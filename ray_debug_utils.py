import ray
import signal
import sys
import atexit
from contextlib import contextmanager
import psutil
import subprocess

class RayDebugManager:
    def __init__(self):
        self.ray_initialized = False
        self.cleanup_registered = False
        
    def force_cleanup(self):
        """Nuclear option - kill all Ray processes"""
        print("🔥 Force cleaning Ray processes...")
        try:
            # Kill ray processes
            subprocess.run(["pkill", "-f", "ray::"], check=False)
            subprocess.run(["pkill", "-f", "raylet"], check=False)
            subprocess.run(["pkill", "-f", "gcs_server"], check=False)
            subprocess.run(["ray", "stop", "--force"], check=False)
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def safe_ray_init(self, **kwargs):
        """Safe Ray initialization with cleanup handlers"""
        if not self.cleanup_registered:
            # Register cleanup on exit
            atexit.register(self.force_cleanup)
            signal.signal(signal.SIGINT, lambda s, f: self._signal_handler())
            signal.signal(signal.SIGTERM, lambda s, f: self._signal_handler())
            self.cleanup_registered = True
        
        if ray.is_initialized():
            print("⚠️  Ray already initialized, shutting down first...")
            ray.shutdown()
            
        print("🚀 Initializing Ray...")
        ray.init(**kwargs)
        self.ray_initialized = True
        return ray
    
    def safe_ray_shutdown(self):
        """Safe Ray shutdown"""
        if self.ray_initialized and ray.is_initialized():
            print("🛑 Shutting down Ray...")
            ray.shutdown()
            self.ray_initialized = False
    
    def _signal_handler(self):
        """Handle Ctrl+C gracefully"""
        print("\n🚨 Interrupt received, cleaning up...")
        self.safe_ray_shutdown()
        self.force_cleanup()
        sys.exit(0)

# Global instance
ray_debug = RayDebugManager()

@contextmanager 
def ray_session(**ray_kwargs):
    """Context manager for Ray sessions - auto cleanup guaranteed"""
    try:
        ray_instance = ray_debug.safe_ray_init(**ray_kwargs)
        yield ray_instance
    finally:
        ray_debug.safe_ray_shutdown()

@contextmanager
def ray_actor_session(actor_class, *args, **kwargs):
    """Context manager for Ray actors - auto cleanup guaranteed"""
    actor = None
    try:
        actor = actor_class.remote(*args, **kwargs)
        yield actor
    finally:
        if actor:
            try:
                ray.kill(actor)
            except:
                pass

# Quick debugging functions
def kill_all_ray():
    """One-liner to kill everything Ray-related"""
    ray_debug.force_cleanup()

def ray_status():
    """Quick Ray cluster status"""
    if ray.is_initialized():
        print("✅ Ray is initialized")
        print(f"📊 Cluster resources: {ray.cluster_resources()}")
        print(f"🔢 Available resources: {ray.available_resources()}")
    else:
        print("❌ Ray is not initialized") 