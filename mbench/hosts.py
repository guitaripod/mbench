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

    def server_info(self):
        return swap.server_info(self.profile.id)

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
        status = self.device.load(profiles.phone_request(self.profile.id, self.profile.phone or {}))
        return status.get("load_seconds")

    def unload(self):
        self.device.unload()

    def server_info(self):
        return swap.direct_info(self.device.server_url)

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


def for_profile(profile):
    return PhoneHost(profile) if profile.phone else GpuHost(profile)
