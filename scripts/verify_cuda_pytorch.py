#!/usr/bin/env python3
"""
CUDA and PyTorch Verification Script for Glitcher
Verifies that CUDA 12.8.0 and PyTorch 2.7.1 are working correctly.
"""

import sys
import platform
import subprocess
from typing import Dict, Any


def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }


def check_pytorch() -> Dict[str, Any]:
    """Check PyTorch installation and version."""
    try:
        import torch

        info = {
            "installed": True,
            "version": torch.__version__,
            "expected_version": "2.7.1",
            "version_match": torch.__version__.startswith("2.7.1"),
            "install_path": torch.__file__,
        }

        # Check compilation info
        try:
            info["build_info"] = torch.version.git_version
        except:
            info["build_info"] = "Not available"

        return info
    except ImportError as e:
        return {
            "installed": False,
            "error": str(e),
            "expected_version": "2.7.1",
            "version_match": False,
        }


def check_cuda() -> Dict[str, Any]:
    """Check CUDA availability and version."""
    cuda_info = {
        "pytorch_cuda_available": False,
        "cuda_version": None,
        "expected_cuda_version": "12.8",
        "cuda_version_compatible": False,
        "device_count": 0,
        "devices": [],
        "current_device": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            cuda_info["pytorch_cuda_available"] = True
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["current_device"] = torch.cuda.current_device()

            # Check if CUDA version is compatible
            if cuda_info["cuda_version"]:
                major_version = cuda_info["cuda_version"].split('.')[0]
                minor_version = cuda_info["cuda_version"].split('.')[1]
                if major_version == "12" and int(minor_version) >= 8:
                    cuda_info["cuda_version_compatible"] = True

            # Get device information
            for i in range(cuda_info["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": device_props.name,
                    "total_memory": device_props.total_memory,
                    "major": device_props.major,
                    "minor": device_props.minor,
                    "multi_processor_count": device_props.multi_processor_count,
                }
                cuda_info["devices"].append(device_info)

    except Exception as e:
        cuda_info["error"] = str(e)

    return cuda_info


def check_nvidia_smi() -> Dict[str, Any]:
    """Check nvidia-smi output."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpus.append({
                            "name": parts[0],
                            "memory_mb": parts[1],
                            "driver_version": parts[2],
                        })

            return {
                "available": True,
                "gpus": gpus,
                "raw_output": result.stdout,
            }
        else:
            return {
                "available": False,
                "error": result.stderr,
            }

    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timeout"}
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except Exception as e:
        return {"available": False, "error": str(e)}


def check_bitsandbytes() -> Dict[str, Any]:
    """Check BitsAndBytes installation and CUDA compatibility."""
    try:
        import bitsandbytes as bnb

        info = {
            "installed": True,
            "version": bnb.__version__,
            "cuda_setup_status": None,
        }

        # Try to check CUDA setup
        try:
            from bitsandbytes.cuda_setup.main import get_cuda_lib_handle, get_cuda_version_string
            info["cuda_setup_status"] = "Available"
            try:
                cuda_version = get_cuda_version_string()
                info["cuda_version_detected"] = cuda_version
            except:
                info["cuda_version_detected"] = "Could not detect"
        except ImportError:
            info["cuda_setup_status"] = "Import failed"
        except Exception as e:
            info["cuda_setup_status"] = f"Error: {str(e)}"

        return info

    except ImportError as e:
        return {
            "installed": False,
            "error": str(e),
        }


def test_basic_operations() -> Dict[str, Any]:
    """Test basic PyTorch and CUDA operations."""
    tests = {
        "cpu_tensor_creation": False,
        "gpu_tensor_creation": False,
        "gpu_computation": False,
        "memory_allocation": False,
    }

    errors = {}

    try:
        import torch

        # Test CPU tensor creation
        try:
            cpu_tensor = torch.randn(100, 100)
            tests["cpu_tensor_creation"] = True
        except Exception as e:
            errors["cpu_tensor_creation"] = str(e)

        if torch.cuda.is_available():
            try:
                # Test GPU tensor creation
                gpu_tensor = torch.randn(100, 100, device='cuda')
                tests["gpu_tensor_creation"] = True

                # Test GPU computation
                result = torch.matmul(gpu_tensor, gpu_tensor.T)
                tests["gpu_computation"] = True

                # Test memory allocation info
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                tests["memory_allocation"] = True

            except Exception as e:
                if "gpu_tensor_creation" not in errors:
                    errors["gpu_tensor_creation"] = str(e)
                if "gpu_computation" not in errors:
                    errors["gpu_computation"] = str(e)
                if "memory_allocation" not in errors:
                    errors["memory_allocation"] = str(e)

    except Exception as e:
        errors["pytorch_import"] = str(e)

    return {"tests": tests, "errors": errors}


def format_memory(bytes_val: int) -> str:
    """Format memory in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def print_verification_report():
    """Print comprehensive verification report."""
    print("=" * 80)
    print("GLITCHER CUDA & PYTORCH VERIFICATION REPORT")
    print("=" * 80)

    # System Information
    print("\n📋 SYSTEM INFORMATION:")
    print("-" * 40)
    system_info = get_system_info()
    print(f"Python Version: {system_info['python_version'].split()[0]}")
    print(f"Platform: {system_info['platform']}")
    print(f"Architecture: {system_info['architecture'][0]}")

    # PyTorch Check
    print("\n🔥 PYTORCH VERIFICATION:")
    print("-" * 40)
    pytorch_info = check_pytorch()
    if pytorch_info["installed"]:
        status = "✅" if pytorch_info["version_match"] else "⚠️"
        print(f"{status} PyTorch Version: {pytorch_info['version']}")
        print(f"   Expected: {pytorch_info['expected_version']}")
        print(f"   Version Match: {pytorch_info['version_match']}")
        if pytorch_info.get("build_info"):
            print(f"   Build Info: {pytorch_info['build_info']}")
    else:
        print(f"❌ PyTorch Installation: FAILED")
        print(f"   Error: {pytorch_info.get('error', 'Unknown error')}")

    # CUDA Check
    print("\n🚀 CUDA VERIFICATION:")
    print("-" * 40)
    cuda_info = check_cuda()

    if cuda_info["pytorch_cuda_available"]:
        status = "✅" if cuda_info["cuda_version_compatible"] else "⚠️"
        print(f"✅ CUDA Available: Yes")
        print(f"{status} CUDA Version: {cuda_info['cuda_version']}")
        print(f"   Expected: {cuda_info['expected_cuda_version']}.x")
        print(f"   Compatible: {cuda_info['cuda_version_compatible']}")
        print(f"✅ Device Count: {cuda_info['device_count']}")
        print(f"   Current Device: {cuda_info['current_device']}")

        if cuda_info["devices"]:
            print("\n   📱 GPU DEVICES:")
            for device in cuda_info["devices"]:
                memory_gb = device["total_memory"] / (1024**3)
                print(f"   Device {device['id']}: {device['name']}")
                print(f"     Memory: {memory_gb:.1f} GB")
                print(f"     Compute: {device['major']}.{device['minor']}")
                print(f"     Processors: {device['multi_processor_count']}")
    else:
        print(f"❌ CUDA Available: No")
        if "error" in cuda_info:
            print(f"   Error: {cuda_info['error']}")

    # NVIDIA-SMI Check
    print("\n🖥️  NVIDIA-SMI VERIFICATION:")
    print("-" * 40)
    nvidia_info = check_nvidia_smi()
    if nvidia_info["available"]:
        print("✅ nvidia-smi: Available")
        if nvidia_info.get("gpus"):
            for i, gpu in enumerate(nvidia_info["gpus"]):
                print(f"   GPU {i}: {gpu['name']}")
                print(f"     Memory: {gpu['memory_mb']} MB")
                print(f"     Driver: {gpu['driver_version']}")
    else:
        print("❌ nvidia-smi: Not Available")
        print(f"   Error: {nvidia_info.get('error', 'Unknown error')}")

    # BitsAndBytes Check
    print("\n🔢 BITSANDBYTES VERIFICATION:")
    print("-" * 40)
    bnb_info = check_bitsandbytes()
    if bnb_info["installed"]:
        print(f"✅ BitsAndBytes Version: {bnb_info['version']}")
        print(f"   CUDA Setup: {bnb_info.get('cuda_setup_status', 'Unknown')}")
        if bnb_info.get("cuda_version_detected"):
            print(f"   CUDA Detected: {bnb_info['cuda_version_detected']}")
    else:
        print("❌ BitsAndBytes: Not Installed")
        print(f"   Error: {bnb_info.get('error', 'Unknown error')}")

    # Basic Operations Test
    print("\n🧪 BASIC OPERATIONS TEST:")
    print("-" * 40)
    test_results = test_basic_operations()

    for test_name, passed in test_results["tests"].items():
        status = "✅" if passed else "❌"
        formatted_name = test_name.replace("_", " ").title()
        print(f"{status} {formatted_name}: {'PASS' if passed else 'FAIL'}")

        if not passed and test_name in test_results["errors"]:
            print(f"     Error: {test_results['errors'][test_name]}")

    # Memory Information
    try:
        import torch
        if torch.cuda.is_available():
            print("\n💾 GPU MEMORY STATUS:")
            print("-" * 40)
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory

                print(f"   GPU {i}:")
                print(f"     Allocated: {format_memory(allocated)}")
                print(f"     Reserved: {format_memory(reserved)}")
                print(f"     Total: {format_memory(total)}")
                print(f"     Free: {format_memory(total - reserved)}")
    except:
        pass

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY:")
    print("=" * 80)

    # Overall status
    pytorch_ok = pytorch_info["installed"] and pytorch_info["version_match"]
    cuda_ok = cuda_info["pytorch_cuda_available"] and cuda_info["cuda_version_compatible"]
    nvidia_ok = nvidia_info["available"]
    bnb_ok = bnb_info["installed"]

    all_tests_passed = all(test_results["tests"].values())

    if pytorch_ok and cuda_ok and nvidia_ok and bnb_ok and all_tests_passed:
        print("🎉 ALL SYSTEMS GO! Your Glitcher environment is ready for GPU acceleration.")
        print("   ✅ PyTorch 2.7.1 with CUDA 12.8 support")
        print("   ✅ GPU(s) detected and accessible")
        print("   ✅ BitsAndBytes for quantization support")
        print("   ✅ All basic operations working")
    else:
        print("⚠️  SOME ISSUES DETECTED:")
        if not pytorch_ok:
            print("   ❌ PyTorch installation or version mismatch")
        if not cuda_ok:
            print("   ❌ CUDA not available or version incompatible")
        if not nvidia_ok:
            print("   ❌ NVIDIA driver/tools not accessible")
        if not bnb_ok:
            print("   ❌ BitsAndBytes not available")
        if not all_tests_passed:
            print("   ❌ Some basic operations failed")

        print("\n🔧 RECOMMENDED ACTIONS:")
        if not pytorch_ok:
            print("   • Reinstall PyTorch 2.7.1 with CUDA 12.8 support")
        if not cuda_ok:
            print("   • Update NVIDIA drivers to support CUDA 12.8+")
            print("   • Verify NVIDIA Container Toolkit installation")
        if not bnb_ok:
            print("   • Install BitsAndBytes: pip install bitsandbytes")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_verification_report()
