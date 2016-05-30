"""
Preliminary wrapping of NEURON version of GIF neuron model to work with PyNN.

Author: Andrew Davison
"""

import numpy
from neuron import h
import efel
from pyNN.neuron import NativeCellType, simulator, run


def simulate_seed(population, I, V0, seed, passive_axon=False):

    """
    Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
    V0 indicate the initial condition V(0)=V0.
    The function returns:
    - time     : ms, support for V, eta_sum, V_T, spks
    - V        : mV, membrane potential
    - eta_sum  : nA, adaptation current
    - V_T      : mV, firing threshold
    - spks     : ms, list of spike times
    """
    assert passive_axon is False
    self = population[0]._cell

    # Input parameters
    T = len(I)*population._simulator.state.dt

    rndd = h.Random(seed)
    rndd.uniform(0, 1)
    self.gif_fun.setRNG(rndd)

    rec_t = h.Vector()
    rec_t.record(h._ref_t)
    h.celsius = 34

    population.initialize(v=V0)
    run(T)

    time = numpy.array(rec_t)
    data = population.get_data().segments[0]
    signals = {}
    for sig in data.analogsignalarrays:
        signals[sig.name] = sig.magnitude

    V = signals['v']
    eta_sum = signals['i_eta']
    urand = signals['rand']
    V_T = signals['gamma1'] + signals['gamma2'] + signals['gamma3'] + self.vt_star

    i = I

    trace = {}
    trace['T'] = time
    trace['V'] = V
    trace['stim_start'] = [0]
    trace['stim_end'] = [T]
    traces = [trace]

    efel_result = efel.getFeatureValues(traces, ['peak_time'])[0]
    spks = efel_result['peak_time']

    return (time[:-1], V[:-1], eta_sum[:-1], V_T[:-1], spks, urand[:-1], i)
