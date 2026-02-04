#!/usr/bin/env python
"""
This module implements the class EnergyMeter, to measure the energy consumption of Python 
functions or code chunks, segregating their energy usage per component (CPU, DRAM, GPU and 
Hard Disk). 
"""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyRAPL
from pynvml import *

import subprocess
import os
import shlex
import json
import time
import threading


class ThreadGpuSamplingPyNvml(threading.Thread):
    """
    Thread to sample GPU power draw and utilization using PyNVML.
    """
    
    def __init__(self, name, gpu_index=0, seconds_between_samples=0.1):
        """
        Initialize the sampling thread.
        
        Args:
            name: Thread name
            gpu_index: GPU device index (default 0)
            seconds_between_samples: Sampling interval in seconds
        """
        threading.Thread.__init__(self)
        self.name = name
        self.daemon = True  # Thread dies when main thread exits
        self.stop = False
        self.gpu_index = gpu_index
        self.seconds_between_samples = seconds_between_samples
        
        # Thread-safe storage with timestamps
        self._lock = threading.Lock()
        self.samples = []  # List of dicts with all measurements
        
        # For backward compatibility
        self.power_draw_history = []
        self.activity_history = []
        
        # Initialize NVML and get GPU handle
        self._nvml_initialized = False
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml_initialized = True
        except NVMLError as e:
            print(f"Warning: Failed to initialize NVML: {e}")
            print("GPU energy measurements will not be available.")
    
    def run(self):
        """
        Sample GPU metrics until self.stop is set to True.
        """
        if not self._nvml_initialized:
            print("NVML not initialized, GPU sampling thread exiting.")
            return
            
        while not self.stop:
            try:
                timestamp = time.time()
                
                # Get power draw (in milliwatts, convert to watts)
                power_mw = nvmlDeviceGetPowerUsage(self.gpu_handle)
                power_w = power_mw / 1000.0
                
                # Get GPU utilization (percentage)
                utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = utilization.gpu
                
                # Get temperature
                temp = nvmlDeviceGetTemperature(self.gpu_handle, NVML_TEMPERATURE_GPU)
                
                # Store sample
                sample = {
                    'timestamp': timestamp,
                    'power_w': power_w,
                    'gpu_util': gpu_util,
                    'temp_c': temp,
                }
                
                with self._lock:
                    self.samples.append(sample)
                    # Update backward compatibility lists
                    self.power_draw_history.append(power_w)
                    self.activity_history.append(gpu_util)
                
            except NVMLError as e:
                print(f"NVML sampling error: {e}")
                # Continue sampling even if one sample fails
            
            time.sleep(self.seconds_between_samples)
    
    def get_samples(self):
        """
        Get a copy of all samples (thread-safe).
        
        Returns:
            list: List of sample dictionaries
        """
        with self._lock:
            return self.samples.copy()
    
    def clear_samples(self):
        """
        Clear all stored samples (useful between experiments).
        """
        with self._lock:
            self.samples.clear()
            self.power_draw_history.clear()
            self.activity_history.clear()
    
    def shutdown(self):
        """
        Stop sampling and cleanup NVML.
        """
        self.stop = True
        self.join(timeout=1.0)  # Wait for thread to finish
        if self._nvml_initialized:
            try:
                nvmlShutdown()
            except NVMLError:
                pass  # Already shutdown
    
    def get_statistics(self):
        """
        Calculate summary statistics from samples.
        
        Returns:
            dict: Mean, max, min, std for power and utilization
        """
        with self._lock:
            if not self.samples:
                return None
            
            power_values = np.array([s['power_w'] for s in self.samples])
            util_values = np.array([s['gpu_util'] for s in self.samples])
            temp_values = np.array([s['temp_c'] for s in self.samples])
            timestamps = np.array([s['timestamp'] for s in self.samples])
            
            # Calculate energy using trapezoidal integration
            energy_j = np.trapezoid(power_values, timestamps)
            energy_wh = energy_j / 3600
            
            return {
                'power_w': {
                    'mean': np.mean(power_values),
                    'max': np.max(power_values),
                    'min': np.min(power_values),
                    'std': np.std(power_values),
                },
                'gpu_util': {
                    'mean': np.mean(util_values),
                    'max': np.max(util_values),
                    'min': np.min(util_values),
                },
                'temp_c': {
                    'mean': np.mean(temp_values),
                    'max': np.max(temp_values),
                    'min': np.min(temp_values),
                },
                'duration_s': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
                'num_samples': len(self.samples),
                'energy_wh': energy_wh,
                'energy_j': energy_j,
            }


class EnergyMeter:
    """
    The consumption of each component is measured as follows:

    - CPU: the energy consumption of the CPU is measured with RAPL via the pyRAPL
        library. RAPL is an API from Intel, which is also semi-compatible with
        AMD. RAPL on Intel has been shown to be accurate thanks to the usage of
        embedded sensors in the processor and memory while AMD uses performance
        counters and is therefore, not so accurate.

    - DRAM: the energy used by the memory is also measure with RAPL via pyRAPL.
        This might not be available for AMD processors and pre-Haswell Intel
        processors. You can find more info here:
        https://dl.acm.org/doi/pdf/10.1145/2989081.2989088.

    - GPU: we measure the energy consumption of the GPU with PyNVML. For this, we
        run a separate thread that samples the power draw of the GPU while the meter
        is running. We then calculate the total energy using trapezoidal integration
        of the power samples over time.

    - Disk: we cannot directly measure the energy consumption of the disk in the same
        way that we do for the other components, so we have implemented an bpftrace
        probe that tracks all the bytes read and written to disk. This probe is run 
        as a separate thread inside EnergyMeter. We then calculate the energy 
        consumption with the following formulae:
        disk_active_time = (bytes_read + bytes_written) / DISK_SPEED
        disk_idle_time = total_meter_time - disk_active_time
        total_energy = disk_active_time * DISK_ACTIVE_POWER +
                        disk_idle_time * DISK_IDLE_POWER
        Note that you are required the provide the parameters DISK_SPEED,
        DISK_ACTIVE_POWER and DISK_IDLE_POWER.
    """

    #################################### CONSTANTS ####################################
    SCRIPT = (
        "tracepoint:syscalls:sys_enter_write {@wbytes[comm] = sum(args->count);} "
        "tracepoint:syscalls:sys_enter_read {@rbytes[comm] = sum(args->count);}"
    )
    ###################################################################################

    def __init__(self, disk_avg_speed=None, disk_active_power=None, disk_idle_power=None, 
                 label=None, include_idle=False, ignore_disk=False, 
                 gpu_index=0, gpu_sampling_rate=0.1):
        """Initiates the variables required to meter the energy consumption of all
        components and sets up the pyRAPL library.
        :param disk_avg_speed: the average read and write speed of the hard disk where
            the code will be run. We recommend measuring this with a speed test such as
            this: https://cloud.google.com/compute/docs/disks/benchmarking-pd-performance.
        :param disk_active_power: the power used by the disk when active. This information
            is usually included in the disk specs. Hint: run lshw to get info about the
            host's disk.
        :param disk_idle_power: the average power used by the disk when idle. Just as for
            disk_active_power, this is usually included in the disk specs.
        :param label: this is just an optional string to identify the meter.
        :param include_idle: if energy used during idle times should be included for disk and 
            GPU.
        :param ignore_disk: False by default, when set to True, disk will not be tracked (used
            for compatibility with systems without access to sudo or bpftrace)
        :param gpu_index: GPU device index to monitor (default 0)
        :param gpu_sampling_rate: sampling frequency in seconds (default 0.1 = 10Hz)
        """
        if label:
            self.label = label
        else:
            self.label = "Meter"

        self.include_idle = include_idle

        # Setup pyRAPL to measure CPU and DRAM.
        try:
            pyRAPL.setup()

            # Create the pyRAPL meter to measure CPU and DRAM energy consumption.
            self.meter = pyRAPL.Measurement(self.label)
        except Exception as e:
            print(f"Warning: Could not initialize pyRAPL: {e}")
            print("CPU and DRAM energy measurements will not be available.")
            self.meter = None

        # Setup disk parameters.
        self.ignore_disk = ignore_disk
        if ignore_disk == False and (disk_avg_speed is None or disk_active_power is None or disk_idle_power is None):
            raise Exception("disk_avg_speed, disk_active_power, and disk_idle_power are necessary values if disk energy will be monitored; if you want to ignore the disk, set ignore_disk=True when calling init.")
        else:
            self.disk_avg_speed = disk_avg_speed
            self.disk_active_power = disk_active_power
            self.disk_idle_power = disk_idle_power

        # Create thread for sampling the power draw of the GPU with improved PyNVML implementation
        self.thread_gpu = ThreadGpuSamplingPyNvml(
            name="GPU Sampling Thread",
            gpu_index=gpu_index,
            seconds_between_samples=gpu_sampling_rate
        )

        # Create command for bpftrace subprocess that will count the bytes read and
        # written to disk.
        self.bpftrace_command = shlex.split(
            "sudo bpftrace -f json -e '{}'".format(EnergyMeter.SCRIPT)
        )

    def begin(self):
        """Begin measuring the energy consumption. This sets the starting datetime and
        reads the current RAPL counters.
        """
        self.start_time = time.time()
        
        # pyRAPL for CPU and DRAM.
        if self.meter:
            self.meter.begin()
        else:
            print("RAPL is not accessible, no CPU or memory energy metrics are available!")

        # bpftrace for disk.
        if not self.ignore_disk:
            try:
                self.popen = subprocess.Popen(
                    self.bpftrace_command, stdout=subprocess.PIPE, preexec_fn=os.setpgrp
                )
                self.bpftrace_pid = os.getpgid(self.popen.pid)
            except Exception as e:
                print(f"Warning: Could not start bpftrace for disk monitoring: {e}")
                self.ignore_disk = True

        # Thread for GPU.
        self.thread_gpu.start()

    def end(self):
        """Finish the measurements and calculate results for CPU and DRAM. This sets the
        duration of the meter and reads again the RAPL counters, calculating how much energy
        was used since the meter began.
        """
        # PyRAPL.
        if self.meter:
            self.meter.end()

        # Kill bpftrace subprocess.
        if not self.ignore_disk:
            try:
                subprocess.check_output(shlex.split("sudo kill {}".format(self.bpftrace_pid)))
            except Exception as e:
                print(f"Warning: Could not stop bpftrace: {e}")

        # Stop tracking GPU power usage.
        self.thread_gpu.stop = True
        self.thread_gpu.join(timeout=2.0)

        # Process bpftrace output.
        if not self.ignore_disk:
            try:
                po = self.popen.stdout.read()
                self.total_rbytes, self.total_wbytes = self.__preprocess_bpftrace_output(po)
            except Exception as e:
                print(f"Warning: Could not read bpftrace output: {e}")
                self.total_rbytes, self.total_wbytes = 0, 0
        else:
            self.total_rbytes, self.total_wbytes = 0, 0

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def __preprocess_bpftrace_output(self, bpftrace_output):
        """Preprocess the output of out bpftrace script and extract the bytes read and
        written. Attention: This must be changed if the bpftrace script changes!
        :param bpftrace_output: the output of the bpftrace script.
        :returns: total_rbytes (float), total_wbytes (float).
        """
        bpftrace_output = bpftrace_output.decode()
        if len(bpftrace_output.strip()) > 0:
            po = bpftrace_output.split("\n")
            rbytes = json.loads(po[3]).get("data").get("@rbytes")
            wbytes = json.loads(po[4]).get("data").get("@wbytes")
            # Do we want to measure other programs disk IO too?
            total_rbytes = rbytes.get("python", 0) + rbytes.get("python3", 0)
            total_wbytes = wbytes.get("python", 0) + wbytes.get("python3", 0)
        else:
            # bpftrace produced no output, which means there was no IO activity in the
            # disk. This only happens when the code run has a very short duration.
            total_rbytes = 0
            total_wbytes = 0

        return total_rbytes, total_wbytes

    def get_total_joules_disk(self):
        """We calculate the disk's energy consumption while the meter was running. For this,
        we require the csv file that was generated by running the bash script start_meters.sh.
        In this case, we utilize the speed and energy consumption parameters given when this
        object was initiated to estimate the disk's energy consumption. The formula used here
        was derived from:
        [1] Kansal, A., Zhao, F., Liu, J., Kothari, N., & Bhattacharya, A. A. (2010, June). 
        Virtual machine power metering and provisioning. In Proceedings of the 1st ACM 
        symposium on Cloud computing (pp. 39-50).

        :returns: the total joules used by the disk between meter.begin() and meter.end().
        """
        if self.ignore_disk:
            return 0
            
        tot_bytes = self.total_rbytes + self.total_wbytes

        # disk_active_time (in seconds) = (bytes_read + bytes_written) / DISK_SPEED
        disk_active_time = tot_bytes / self.disk_avg_speed

        # disk_idle_time (in seconds) = total_meter_time - disk_active_time
        disk_idle_time = self.duration - disk_active_time

        # total_energy = disk_active_time * DISK_ACTIVE_POWER (+ disk_idle_time * DISK_IDLE_POWER)
        te = disk_active_time * self.disk_active_power
        if self.include_idle:
            te += disk_idle_time * self.disk_idle_power
        return te

    def get_total_joules_cpu(self):
        """We obtain the total joules consumed by the CPU from pyRAPL.
        :returns: the total joules used by the CPU between meter.begin() and meter.end().
        """
        # pyRAPL returns the microjoules, so we convert them to joules.
        if self.meter and self.meter.result.pkg:
            return np.array(self.meter.result.pkg) * 1e-6
        else:
            print("RAPL did not record energy for pkg!")
            return np.array([0])

    def get_total_joules_dram(self):
        """We obtain the total joules consumed by the DRAM from pyRAPL.
        :returns: the total joules used by the DRAM between meter.begin() and meter.end().
        """
        # pyRAPL returns the microjoules, so we convert them to joules.
        if self.meter and self.meter.result.dram:
            return np.array(self.meter.result.dram) * 1e-6
        else:
            print("RAPL did not record energy for dram!")
            return np.array([0])

    def get_total_joules_gpu(self):
        """We calculate the GPU's energy consumption while the meter was running using
        trapezoidal integration of power samples over time. This provides more accurate
        energy measurements than the previous mean power × duration approach.
        
        :returns: the total joules used by the GPU between meter.begin() and meter.end().
        """
        stats = self.thread_gpu.get_statistics()
        
        if stats is None or stats['num_samples'] == 0:
            print("Warning: No GPU samples collected!")
            return 0

        if not self.include_idle:
            # Filter to only active samples (GPU utilization > 0)
            samples = self.thread_gpu.get_samples()
            active_samples = [s for s in samples if s['gpu_util'] > 0]
            
            if len(active_samples) == 0:
                return 0
            
            # Recalculate energy for active periods only
            timestamps = np.array([s['timestamp'] for s in active_samples])
            power_values = np.array([s['power_w'] for s in active_samples])
            
            energy_j = np.trapezoid(power_values, timestamps)
            return energy_j
        else:
            # Use total energy (including idle)
            return stats['energy_j']

    def get_total_joules_per_component(self):
        """This returns the total energy consumption in joules between meter.begin() and
        meter.end() segregated by component (CPU, DRAM, GPU and disk).
        :returns: a dictionary with the total joules used by each component.
        """
        cpu = self.get_total_joules_cpu()
        dram = self.get_total_joules_dram()
        gpu = self.get_total_joules_gpu()
        disk = self.get_total_joules_disk()
        res = {
            "cpu": cpu,
            "dram": dram,
            "gpu": gpu,
            "disk": disk,
        }
        return res
    
    def get_gpu_statistics(self):
        """
        Get detailed GPU statistics including temperature and utilization.
        
        :returns: dictionary with GPU statistics or None if no samples
        """
        return self.thread_gpu.get_statistics()

    def plot_total_joules_per_component(self, include_total=True):
        """This plots the total energy consumption in joules between meter.begin() and
        meter.end() and the total consumption by each component (CPU, DRAM, GPU and disk).
        """
        data = self.get_total_joules_per_component()
        if include_total:
            data["total"] = (
                np.sum(data.get("cpu"))
                + np.sum(data.get("dram"))
                + data.get("disk")
                + data.get("gpu")
            )
        keys = data.keys()
        values = [float(val) for val in data.values()]

        fig, ax = plt.subplots()
        bars = ax.bar(list(keys), values)
        ax.bar_label(bars)
        plt.xlabel("Components")
        plt.ylabel("Joules")
        plt.title(self.label)

        plt.show()
    
    def cleanup(self):
        """
        Cleanup resources (call this when completely done with the meter).
        """
        self.thread_gpu.shutdown()