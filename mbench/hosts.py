import urllib.error
import urllib.request

from . import gpu, paths, phone, profiles, stack, swap


class GpuHost:
    """The desktop: llama-swap in front of the card, with the driver reporting power, memory and faults."""

    kind = "gpu"

    def __init__(self, profile):
        self.profile = profile

    def base_url(self):
        return paths.SWAP_URL

    def reachable(self):
        return swap.reachable()

    def ensure_loaded(self):
        return swap.ensure_loaded(self.profile.id)

    def unload(self):
        swap.unload()

    def report(self, progress):
        return None

    def server_info(self):
        return swap.server_info(self.profile.id)

    def kv_unified(self):
        """Whether the server shares one cache across its slots, which decides how much work can be in flight."""
        return any(flag in (self.profile.cmd or "") for flag in ("-kvu", "--kv-unified"))

    def describe(self):
        return gpu.describe()

    def build(self, info):
        return stack.build(self.profile.engine, info)

    def digest(self, info):
        """The weights llama-swap actually served, read off /props and hashed here."""
        payload = (info or {}).get("info") or {}
        return stack.digest_of(payload.get("model_path"))

    def sampler(self, interval_ms=1000):
        return gpu.Sampler(interval_ms=interval_ms)

    def contention(self, samples=3):
        return gpu.contention(samples=samples)

    def describe_contention(self, found):
        return gpu.describe_contention(found)

    def cooldown(self, report=None):
        """A card measures the same whether it just finished a run or not; a phone does not."""
        return {}

    def guard(self, halt, events):
        return None


class PhoneHost:
    """A phone running mbenchd: the same llama.cpp, built for arm64 iOS, with the app reporting thermal state,
    memory footprint and battery in place of the driver's power and VRAM."""

    kind = "phone"

    def __init__(self, profile):
        self.profile = profile
        settings = profile.phone or {}
        self.device = phone.Device(settings.get("control_url") or phone.CONTROL_URL,
                                   settings.get("server_url") or phone.SERVER_URL)

    def base_url(self):
        return self.device.server_url

    def reachable(self):
        return self.device.reachable()

    def ensure_loaded(self):
        """Starts the app fresh and loads the model. Memory a model held is not always given back when the next one
        replaces it, and an app that has loaded three models in a row is killed part way through measuring the fourth
        — so every run gets a process of its own, which is also the same starting state every time."""
        request = profiles.phone_request(self.profile.id, self.profile.phone or {})
        try:
            phone.launch()
            self.device.cooldown(floor=3, hold=0, cap=60)
        except RuntimeError:
            pass
        try:
            return self.device.load(request).get("load_seconds")
        except phone.Unreachable as first:
            try:
                phone.launch()
            except RuntimeError as error:
                raise phone.Unreachable(f"{first}; could not relaunch the app: {error}") from first
            self.device.cooldown(floor=5, hold=0, cap=120)
            return self.device.load(request).get("load_seconds")

    def unload(self):
        self.device.unload()

    def report(self, progress):
        self.device.report(progress)

    def server_info(self):
        return swap.direct_info(self.device.server_url)

    def kv_unified(self):
        return any(flag in ("-kvu", "--kv-unified") for flag in (self.profile.phone or {}).get("extra_args") or [])

    def describe(self):
        return self.device.describe()

    def build(self, info):
        return {**stack.build(self.profile.engine, info), **self.device.build()}

    def digest(self, info):
        """The phone hashes what it loaded and reports it; the file never leaves the device."""
        return ((self.device.health() or {}).get("server") or {}).get("model_sha256")

    def sampler(self, interval_ms=1000):
        return phone.Sampler(self.device, interval_ms=interval_ms)

    def contention(self, samples=3):
        return []

    def describe_contention(self, found):
        return ""

    def cooldown(self, report=None):
        return self.device.cooldown(report=report)

    def guard(self, halt, events):
        return phone.Guard(self.device, halt, events)


class NullSampler:
    """Nothing to sample: a machine across the network reports no power of its own, so a run records none rather than
    guessing one."""

    def window(self, start, end):
        return {}

    def energy_wh(self, start, end):
        return None

    def timeline(self):
        return []

    def close(self):
        return None


class RemoteHost:
    """An OpenAI-compatible server on another machine — the Mac running MLX. Nothing here is launched or unloaded:
    the server holds the model, and mbench only measures what it answers."""

    kind = "remote"

    def __init__(self, profile):
        self.profile = profile
        self.settings = profile.remote or {}

    def base_url(self):
        return (self.settings.get("base_url") or "").rstrip("/")

    def reachable(self):
        try:
            with urllib.request.urlopen(self.base_url() + "/v1/models", timeout=10):
                return True
        except (OSError, urllib.error.HTTPError):
            return False

    def ensure_loaded(self):
        """The server loads what a request names, so a run only waits for it to answer one."""
        return None

    def report(self, progress):
        return None

    def unload(self):
        return None

    def server_info(self):
        return swap.direct_info(self.base_url())

    def kv_unified(self):
        return False

    def describe(self):
        return {
            "class": self.settings.get("class") or "mac",
            "device": self.settings.get("device") or "remote server",
            "soc": self.settings.get("soc"),
            "ram_gb": self.settings.get("ram_gb"),
            "bandwidth_gbs": self.settings.get("bandwidth_gbs"),
            "os": self.settings.get("os"),
        }

    def build(self, info):
        return stack.build(self.profile.engine, info)

    def digest(self, info):
        return None

    def sampler(self, interval_ms=1000):
        return NullSampler()

    def contention(self, samples=3):
        return []

    def describe_contention(self, found):
        return ""

    def cooldown(self, report=None):
        return {}

    def guard(self, halt, events):
        return None


def for_profile(profile):
    if profile.phone:
        return PhoneHost(profile)
    if profile.remote:
        return RemoteHost(profile)
    return GpuHost(profile)
