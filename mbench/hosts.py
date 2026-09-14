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
        """Loads the model, restarting the app once if it will not answer. An app that was backgrounded mid-run comes
        back with a server it can no longer stop, and a benchmark that gives up there loses a night of measurements
        over something a relaunch fixes."""
        request = profiles.phone_request(self.profile.id, self.profile.phone or {})
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


def for_profile(profile):
    return PhoneHost(profile) if profile.phone else GpuHost(profile)
