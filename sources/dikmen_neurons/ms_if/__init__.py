"""
MS-IF Family: Multi-Scale Integrate-and-Fire for ISAC
Axis: Which time scale and signal component to listen to?
Models: Dual-tau (#12), Chirp (#13), Phase (#14), CrossInhib (#15),
        Doppler (#16), STFT (#17), Beam (#18)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..base import BaseNeuron, spike_hard, spike_stochastic


class DualTauLIF(BaseNeuron):
    """#12 Dual Time-Constant LIF. Fast (comms) + slow (sensing) parallel membranes."""
    _family = "ms_if"
    _description = "Two parallel membranes: fast (symbol rate) + slow (Doppler)."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_beta_fast = nn.Parameter(torch.full((size,), math.log(0.8)))
        self.log_beta_slow = nn.Parameter(torch.full((size,), math.log(0.98)))
        self.mix = nn.Parameter(torch.full((size,), 0.0))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "mem_fast": torch.zeros(batch_size, self.size, device=device),
            "mem_slow": torch.zeros(batch_size, self.size, device=device),
        }

    def single_step(self, x_t, state):
        bf = torch.sigmoid(self.log_beta_fast)
        bs = torch.sigmoid(self.log_beta_slow)
        mf = bf * state["mem_fast"] + x_t
        ms = bs * state["mem_slow"] + x_t
        w = torch.sigmoid(self.mix)
        mem = w * mf + (1 - w) * ms
        spk = spike_hard(mem, self.threshold)
        mf = mf - spk * self.threshold * w
        ms = ms - spk * self.threshold * (1 - w)
        return spk, {"mem": mem, "mem_fast": mf, "mem_slow": ms}


class ChirpLIF(BaseNeuron):
    """#13 Chirp-Resonant LIF. Oscillatory membrane resonates with FMCW chirp rate."""
    _family = "ms_if"
    _description = "Oscillatory membrane: fires on resonance with FMCW chirp."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_omega = nn.Parameter(torch.randn(size) * 0.3)
        self.log_A = nn.Parameter(torch.full((size,), -0.5))
        self.phi = nn.Parameter(torch.zeros(size))
        self.log_bw = nn.Parameter(torch.full((size,), math.log(0.3)))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "t": torch.zeros(batch_size, 1, device=device),
        }

    def single_step(self, x_t, state):
        t = state["t"] + 1
        omega = self.log_omega.exp()
        A = self.log_A.exp()
        osc = A * torch.sin(omega * t + self.phi)
        mem = self.beta * state["mem"] + x_t + osc
        spk = spike_hard(mem, self.threshold)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem, "t": t}


class PhaseLIF(BaseNeuron):
    """#14 Complex-Valued Phase LIF. Phase-sensitive spike for OFDM subcarriers."""
    _family = "ms_if"
    _description = "Complex-valued membrane: spike on phase difference exceeding threshold."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.beta_real = nn.Parameter(torch.full((size,), 0.95))
        self.beta_imag = nn.Parameter(torch.full((size,), 0.95))
        self.theta_phase = nn.Parameter(torch.full((size,), math.pi / 4))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "mem_real": torch.zeros(batch_size, self.size, device=device),
            "mem_imag": torch.zeros(batch_size, self.size, device=device),
        }

    def single_step(self, x_t, state):
        mr = torch.sigmoid(self.beta_real) * state["mem_real"] + x_t
        mi = torch.sigmoid(self.beta_imag) * state["mem_imag"] + x_t * 0.1
        envelope = torch.sqrt(mr ** 2 + mi ** 2 + 1e-8)
        phase = torch.atan2(mi, mr)
        mem = envelope
        spk = spike_hard(mem, self.threshold)
        mr = mr * (1 - spk)
        mi = mi * (1 - spk)
        return spk, {"mem": mem, "mem_real": mr, "mem_imag": mi}


class CrossInhibLIF(BaseNeuron):
    """#15 Cross-Inhibitory Pair LIF. Excitatory + inhibitory pair for interference cancellation."""
    _family = "ms_if"
    _description = "Excitatory-inhibitory pair: natural comms-sensing interference suppression."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.w_inh = nn.Parameter(torch.full((size,), 0.3))
        self.w_exc = nn.Parameter(torch.full((size,), 0.3))
        self.beta_e = nn.Parameter(torch.full((size,), 0.95))
        self.beta_i = nn.Parameter(torch.full((size,), 0.90))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "mem_exc": torch.zeros(batch_size, self.size, device=device),
            "mem_inh": torch.zeros(batch_size, self.size, device=device),
        }

    def single_step(self, x_t, state):
        be = torch.sigmoid(self.beta_e)
        bi = torch.sigmoid(self.beta_i)
        wi = F.softplus(self.w_inh)
        we = F.softplus(self.w_exc)
        me = be * state["mem_exc"] + x_t - wi * state["mem_inh"]
        mi = bi * state["mem_inh"] + x_t - we * state["mem_exc"]
        mem = F.relu(me)
        spk = spike_hard(mem, self.threshold)
        me = me - spk * self.threshold
        return spk, {"mem": mem, "mem_exc": me, "mem_inh": mi}


class DopplerLIF(BaseNeuron):
    """#16 Doppler-Tuned Adaptive LIF. Adaptive resonance tuned to input Doppler shift."""
    _family = "ms_if"
    _description = "Adaptive resonance: auto-locks to Doppler shift frequency."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.f_center = nn.Parameter(torch.randn(size) * 0.3)
        self.log_bw = nn.Parameter(torch.full((size,), math.log(0.3)))
        self.adapt_rate = nn.Parameter(torch.full((size,), 0.05))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "prev_x": torch.zeros(batch_size, self.size, device=device),
            "f_est": self.f_center.unsqueeze(0).expand(batch_size, -1).clone().to(device),
        }

    def single_step(self, x_t, state):
        freq_proxy = (x_t - state["prev_x"]).abs()
        rate = torch.sigmoid(self.adapt_rate)
        f_est = (1 - rate) * state["f_est"] + rate * freq_proxy
        bw = self.log_bw.exp()
        gain = torch.exp(-0.5 * ((freq_proxy - self.f_center) / bw) ** 2)
        mem = self.beta * state["mem"] + x_t * gain
        spk = spike_hard(mem, self.threshold)
        mem = mem * (1.0 - spk)
        return spk, {"mem": mem, "prev_x": x_t, "f_est": f_est}


class STFTIF(BaseNeuron):
    """#33 STFT-IF. Sliding-window DFT: spike on spectral band energy exceeding threshold."""
    _family = "ms_if"
    _description = "Short-time Fourier: spike on energy in a learned frequency band."

    def __init__(self, size, beta=0.95, threshold=1.0, window_len=16):
        super().__init__(size, beta, threshold)
        self.window_len = window_len
        self.band_center = nn.Parameter(torch.rand(size) * 0.5)
        self.log_band_width = nn.Parameter(torch.full((size,), math.log(0.1)))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "buffer": torch.zeros(batch_size, self.window_len, self.size, device=device),
        }

    def single_step(self, x_t, state):
        buf = torch.cat([state["buffer"][:, 1:, :], x_t.unsqueeze(1)], dim=1)
        N = self.window_len
        freqs = torch.arange(N // 2 + 1, device=x_t.device, dtype=x_t.dtype) / N
        fc = torch.sigmoid(self.band_center)
        bw = self.log_band_width.exp().clamp(min=0.01)

        band_mask = torch.exp(-0.5 * ((freqs.unsqueeze(1) - fc) / bw) ** 2)
        spectrum = torch.fft.rfft(buf, dim=1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2)
        band_energy = (power * band_mask.unsqueeze(0)).sum(dim=1)

        mem = self.beta * state["mem"] + band_energy
        spk = spike_hard(mem, self.threshold)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem, "buffer": buf}


class BeamIF(BaseNeuron):
    """#34 Beamforming Spatial-Filter IF. Spatial filtering via learned steering vector."""
    _family = "ms_if"
    _description = "Beamforming: spatial filter spike for MIMO/ISAC array processing."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.theta_look = nn.Parameter(torch.randn(size) * 0.3)
        self.log_weight = nn.Parameter(torch.zeros(size))

    def single_step(self, x_t, state):
        steering = torch.cos(self.theta_look) * x_t
        weight = torch.sigmoid(self.log_weight)
        beam_out = weight * steering
        mem = self.beta * state["mem"] + beam_out.abs()
        spk = spike_hard(mem, self.threshold)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


REGISTRY = {
    "Dual-tau-LIF": DualTauLIF,
    "Chirp-LIF": ChirpLIF,
    "Phase-LIF": PhaseLIF,
    "CrossInhib-LIF": CrossInhibLIF,
    "Doppler-LIF": DopplerLIF,
    "STFT-IF": STFTIF,
    "Beam-IF": BeamIF,
}
