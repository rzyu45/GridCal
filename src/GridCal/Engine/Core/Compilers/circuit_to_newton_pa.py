# GridCal
# Copyright (C) 2022 Santiago Peñate Vera
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
import os.path
from typing import List, Dict
from GridCal.Engine.basic_structures import Logger
from GridCal.Engine.Core.multi_circuit import MultiCircuit
from GridCal.Engine.basic_structures import BranchImpedanceMode
from GridCal.Engine.basic_structures import BusMode
from GridCal.Engine.Devices.enumerations import ConverterControlType, TransformerControlType
from GridCal.Engine.Devices import *
from GridCal.Engine.basic_structures import Logger, SolverType, ReactivePowerControlMode, TapsControlMode
from GridCal.Engine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from GridCal.Engine.Simulations.PowerFlow.power_flow_results import PowerFlowResults
# from GridCal.Engine.Simulations.OPF.opf_results import OptimalPowerFlowResults
# from GridCal.Engine.Simulations.OPF.opf_options import OptimalPowerFlowOptions, ZonalGrouping
from GridCal.Engine.IO.file_system import get_create_gridcal_folder
import GridCal.Engine.basic_structures as bs

try:
    import newtonpa as npa

    activation = npa.findAndActivateLicense()
    # activate
    if not npa.isLicenseActivated():
        npa_license = os.path.join(get_create_gridcal_folder(), 'newton.lic')
        if os.path.exists(npa_license):
            npa.activateLicense(npa_license)
            if npa.isLicenseActivated():
                NEWTON_PA_AVAILABLE = True
            else:
                print('Newton Power Analytics v' + npa.get_version(),
                      "installed, tried to activate with {} but the license did not work :/".format(npa_license))
                NEWTON_PA_AVAILABLE = False
        else:
            print('Newton Power Analytics v' + npa.get_version(), "installed but not licensed")
            NEWTON_PA_AVAILABLE = False
    else:
        print('Newton Power Analytics v' + npa.get_version())
        NEWTON_PA_AVAILABLE = True

except ImportError as e:
    NEWTON_PA_AVAILABLE = False
    print('Newton Power Analytics is not available:', e)

# numpy integer type for Newton's uword
BINT = np.ulonglong


def add_npa_areas(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", ntime: int=1):

    d = dict()

    for i, area in enumerate(circuit.areas):

        elm = npa.Area(uuid=area.idtag,
                       secondary_id=str(area.code),
                       name=area.name,
                       time_steps=ntime)

        npa_circuit.addArea(elm)

        d[area] = elm

    return d


def add_npa_zones(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", ntime: int = 1):
    d = dict()

    for i, area in enumerate(circuit.zones):
        elm = npa.Zone(uuid=area.idtag,
                       secondary_id=str(area.code),
                       name=area.name,
                       time_steps=ntime)

        npa_circuit.addZone(elm)

        d[area] = elm

    return d


def add_npa_contingency_groups(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", ntime: int = 1):
    d = dict()

    for i, elm in enumerate(circuit.contingency_groups):
        dev = npa.ContingenciesGroup(uuid=elm.idtag,
                                     secondary_id=str(elm.code),
                                     name=elm.name,
                                     time_steps=ntime,
                                     category=elm.category)

        npa_circuit.addContingenciesGroup(dev)

        d[elm] = dev

    return d


def add_npa_contingencies(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", ntime: int = 1,
                          groups_dict=dict()):
    d = dict()

    for i, elm in enumerate(circuit.contingencies):
        dev = npa.Contingency(uuid=elm.idtag,
                              secondary_id=str(elm.code),
                              name=elm.name,
                              time_steps=ntime,
                              device_uuid=elm.device_idtag,
                              prop=elm.prop,
                              value=elm.value,
                              group=groups_dict[elm.group])

        npa_circuit.addContingency(dev)

        d[elm] = dev

    return d


def add_npa_investment_groups(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", ntime: int = 1):
    d = dict()

    for i, elm in enumerate(circuit.investments_groups):
        dev = npa.InvestmentsGroup(uuid=elm.idtag,
                                   secondary_id=str(elm.code),
                                   name=elm.name,
                                   time_steps=ntime,
                                   category=elm.category)

        npa_circuit.addInvestmentsGroup(dev)

        d[elm] = dev

    return d


def add_npa_investments(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", ntime: int = 1,
                        groups_dict=dict()):
    d = dict()

    for i, elm in enumerate(circuit.investments):
        elm = npa.Investment(uuid=elm.idtag,
                             secondary_id=str(elm.code),
                             name=elm.name,
                             time_steps=ntime,
                             device_uuid=elm.device_idtag,
                             group=groups_dict[elm.group])

        npa_circuit.addInvestment(elm)

        d[elm] = elm

    return d


def add_npa_buses(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", time_series: bool, ntime: int=1, tidx=None,
                  area_dict=None):
    """
    Convert the buses to Newton buses
    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise, just the snapshot
    :param ntime: number of time steps
    :return: bus dictionary buses[uuid] -> Bus
    """
    areas_dict = {elm: k for k, elm in enumerate(circuit.areas)}
    bus_dict = dict()

    for i, bus in enumerate(circuit.buses):

        elm = npa.CalculationNode(uuid=bus.idtag,
                                  secondary_id=str(bus.code),
                                  name=bus.name,
                                  time_steps=ntime,
                                  slack=bus.is_slack,
                                  dc=bus.is_dc,
                                  nominal_voltage=bus.Vnom,
                                  vmin=bus.Vmin,
                                  vmax=bus.Vmax,
                                  area=area_dict[bus.area] if bus.area is not None else None)

        if time_series and ntime > 1:
            elm.active = bus.active_prof.astype(BINT) if tidx is None else bus.active_prof.astype(BINT)[tidx]
        else:
            elm.active = np.ones(ntime, dtype=BINT) * int(bus.active)

        npa_circuit.addCalculationNode(elm)
        bus_dict[bus.idtag] = elm

    return bus_dict


def add_npa_loads(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """

    devices = circuit.get_loads()
    for k, elm in enumerate(devices):

        load = npa.Load(uuid=elm.idtag,
                        secondary_id=str(elm.code),
                        name=elm.name,
                        calc_node=bus_dict[elm.bus.idtag],
                        time_steps=ntime,
                        P=elm.P,
                        Q=elm.Q)

        if time_series:
            load.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            load.P = elm.P_prof if tidx is None else elm.P_prof[tidx]
            load.Q = elm.Q_prof if tidx is None else elm.Q_prof[tidx]
            load.cost_1 = elm.Cost_prof if tidx is None else elm.Cost_prof[tidx]
        else:
            load.active = np.ones(ntime, dtype=BINT) * int(elm.active)
            load.setAllCost1(elm.Cost)

        npa_circuit.addLoad(load)


def add_npa_static_generators(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict,
                              time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """
    devices = circuit.get_static_generators()
    for k, elm in enumerate(devices):

        pe_inj = npa.PowerElectronicsInjection(uuid=elm.idtag,
                                               secondary_id=str(elm.code),
                                               name=elm.name,
                                               calc_node=bus_dict[elm.bus.idtag],
                                               time_steps=ntime,
                                               P=elm.P,
                                               Q=elm.Q)

        if time_series:
            pe_inj.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            pe_inj.P = elm.P_prof if tidx is None else elm.P_prof[tidx]
            pe_inj.Q = elm.Q_prof if tidx is None else elm.Q_prof[tidx]
            pe_inj.cost_1 = elm.Cost_prof if tidx is None else elm.Cost_prof[tidx]
        else:
            pe_inj.active = np.ones(ntime, dtype=BINT) * int(elm.active)
            pe_inj.setAllCost1(elm.Cost)

        npa_circuit.addPowerElectronicsInjection(pe_inj)


def add_npa_shunts(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """
    devices = circuit.get_shunts()
    for k, elm in enumerate(devices):

        sh = npa.Capacitor(uuid=elm.idtag,
                           secondary_id=str(elm.code),
                           name=elm.name,
                           calc_node=bus_dict[elm.bus.idtag],
                           time_steps=ntime,
                           G=elm.G,
                           B=elm.B)

        if time_series:
            sh.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            sh.G = elm.G_prof if tidx is None else elm.G_prof[tidx]
            sh.B = elm.B_prof if tidx is None else elm.B_prof[tidx]
        else:
            sh.active = np.ones(ntime, dtype=BINT) * int(elm.active)

        npa_circuit.addCapacitor(sh)


def add_npa_generators(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """
    devices = circuit.get_generators()

    for k, elm in enumerate(devices):

        gen = npa.Generator(uuid=elm.idtag,
                            name=elm.name,
                            calc_node=bus_dict[elm.bus.idtag],
                            time_steps=ntime,
                            P=elm.P,
                            Vset=elm.Vset,
                            Pmin=elm.Pmin,
                            Pmax=elm.Pmax,
                            Qmin=elm.Qmin,
                            Qmax=elm.Qmax,
                            dispatchable_default=BINT(elm.enabled_dispatch)
                            )

        gen.nominal_power = elm.Snom

        if elm.is_controlled:
            gen.setAllControllable(1)
        else:
            gen.setAllControllable(0)

        if time_series:
            gen.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            gen.P = elm.P_prof if tidx is None else elm.P_prof[tidx]
            gen.Vset = elm.Vset_prof if tidx is None else elm.Vset_prof[tidx]
            gen.cost_0 = elm.Cost0_prof if tidx is None else elm.Cost0_prof[tidx]
            gen.cost_1 = elm.Cost_prof if tidx is None else elm.Cost_prof[tidx]
            gen.cost_2 = elm.Cost2_prof if tidx is None else elm.Cost2_prof[tidx]
        else:
            gen.active = np.ones(ntime, dtype=BINT) * int(elm.active)
            gen.P = np.ones(ntime, dtype=float) * elm.P
            gen.Vset = np.ones(ntime, dtype=float) * elm.Vset
            gen.setAllCost0(elm.Cost0)
            gen.setAllCost1(elm.Cost)
            gen.setAllCost2(elm.Cost2)

        npa_circuit.addGenerator(gen)


def get_battery_data(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """
    devices = circuit.get_batteries()

    for k, elm in enumerate(devices):

        gen = npa.Battery(uuid=elm.idtag,
                          name=elm.name,
                          calc_node=bus_dict[elm.bus.idtag],
                          time_steps=ntime,
                          nominal_energy=elm.Enom,
                          P=elm.P,
                          Vset=elm.Vset,
                          soc_max=elm.max_soc,
                          soc_min=elm.min_soc,
                          Qmin=elm.Qmin,
                          Qmax=elm.Qmax,
                          Pmin=elm.Pmin,
                          Pmax=elm.Pmax,
                          )

        gen.nominal_power = elm.Snom
        gen.charge_efficiency = elm.charge_efficiency
        gen.discharge_efficiency = elm.discharge_efficiency

        if elm.is_controlled:
            gen.setAllControllable(1)
        else:
            gen.setAllControllable(0)

        if time_series:
            gen.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            gen.P = elm.P_prof if tidx is None else elm.P_prof[tidx]
            gen.Vset = elm.Vset_prof if tidx is None else elm.Vset_prof[tidx]
            gen.cost_0 = elm.Cost0_prof if tidx is None else elm.Cost0_prof[tidx]
            gen.cost_1 = elm.Cost_prof if tidx is None else elm.Cost_prof[tidx]
            gen.cost_2 = elm.Cost2_prof if tidx is None else elm.Cost2_prof[tidx]
        else:
            gen.active = np.ones(ntime, dtype=BINT) * int(elm.active)
            gen.P = np.ones(ntime, dtype=float) * elm.P
            gen.Vset = np.ones(ntime, dtype=float) * elm.Vset
            gen.setAllCost0(elm.Cost0)
            gen.setAllCost1(elm.Cost)
            gen.setAllCost2(elm.Cost2)

        npa_circuit.addBattery(gen)


def add_npa_line(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """

    # Compile the lines
    for i, elm in enumerate(circuit.lines):
        lne = npa.AcLine(uuid=elm.idtag,
                         secondary_id=str(elm.code),
                         name=elm.name,
                         calc_node_from=bus_dict[elm.bus_from.idtag],
                         calc_node_to=bus_dict[elm.bus_to.idtag],
                         time_steps=ntime,
                         length=elm.length,
                         rate=elm.rate,
                         active_default=elm.active,
                         r=elm.R,
                         x=elm.X,
                         b=elm.B,
                         monitor_loading_default=elm.monitor_loading,
                         monitor_contingency_default=elm.contingency_enabled)

        if time_series:
            lne.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            lne.rates = elm.rate_prof if tidx is None else elm.rate_prof[tidx]
            contingency_rates = elm.rate_prof * elm.contingency_factor
            lne.contingency_rates = contingency_rates if tidx is None else contingency_rates[tidx]
            lne.overload_cost = elm.Cost_prof
        else:
            lne.setAllOverloadCost(elm.Cost)

        npa_circuit.addAcLine(lne)


def get_transformer_data(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict,
                         time_series: bool, ntime=1, tidx=None, override_controls=False):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    :param tidx:
    :param override_controls: If true the controls are set to Fix
    """

    ctrl_dict = {
        TransformerControlType.fixed: npa.BranchControlModes.Fixed,
        TransformerControlType.Pt: npa.BranchControlModes.BranchPt,
        TransformerControlType.Qt: npa.BranchControlModes.BranchQt,
        TransformerControlType.PtQt: npa.BranchControlModes.BranchPt,
        TransformerControlType.Vt: npa.BranchControlModes.BranchVt,
        TransformerControlType.PtVt: npa.BranchControlModes.BranchPt,
    }

    for i, elm in enumerate(circuit.transformers2w):
        tr2 = npa.Transformer2WFull(uuid=elm.idtag,
                                    secondary_id=str(elm.code),
                                    name=elm.name,
                                    calc_node_from=bus_dict[elm.bus_from.idtag],
                                    calc_node_to=bus_dict[elm.bus_to.idtag],
                                    time_steps=ntime,
                                    Vhigh=elm.HV,
                                    Vlow=elm.LV,
                                    rate=elm.rate,
                                    active_default=elm.active,
                                    r=elm.R,
                                    x=elm.X,
                                    g=elm.G,
                                    b=elm.B,
                                    monitor_loading_default=elm.monitor_loading,
                                    monitor_contingency_default=elm.contingency_enabled,
                                    tap=elm.tap_module,
                                    phase=elm.angle)
        if time_series:
            contingency_rates = elm.rate_prof * elm.contingency_factor
            active_prof = elm.active_prof.astype(BINT)

            tr2.active = active_prof if tidx is None else active_prof[tidx]
            tr2.rates = elm.rate_prof if tidx is None else elm.rate_prof[tidx]
            tr2.contingency_rates = contingency_rates if tidx is None else contingency_rates[tidx]
            tr2.tap = elm.tap_module_prof if tidx is None else elm.tap_module_prof[tidx]
            tr2.phase = elm.angle_prof if tidx is None else elm.angle_prof[tidx]
            tr2.overload_cost = elm.Cost_prof
        else:
            tr2.setAllOverloadCost(elm.Cost)

        # control vars
        if override_controls:
            tr2.setAllControlMode(npa.BranchControlModes.Fixed)
        else:
            tr2.setAllControlMode(ctrl_dict[elm.control_mode])  # TODO: Warn about this

        tr2.phase_min = elm.angle_min
        tr2.phase_max = elm.angle_max
        tr2.tap_min = elm.tap_module_min
        tr2.tap_max = elm.tap_module_max
        npa_circuit.addTransformers2wFul(tr2)


def get_vsc_data(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """
    for i, elm in enumerate(circuit.vsc_devices):

        """
        uuid: str = '', 
        secondary_id: str = '', 
        name: str = '', 
        calc_node_from: newtonpa.CalculationNode = None, 
        calc_node_to: newtonpa.CalculationNode = None, 
        cn_from: newtonpa.ConnectivityNode = None, 
        cn_to: newtonpa.ConnectivityNode = None, 
        time_steps: int = 1, 
        active_default: int = 1)
        
        uuid='', secondary_id='', name='', calc_node_from=None, calc_node_to=None, cn_from=None, cn_to=None, time_steps=1, active_default=1
        """

        vsc = npa.AcDcConverter(uuid=elm.idtag,
                                secondary_id=str(elm.code),
                                name=elm.name,
                                calc_node_from=bus_dict[elm.bus_from.idtag],
                                calc_node_to=bus_dict[elm.bus_to.idtag],
                                time_steps=ntime,
                                active_default=elm.active)

        vsc.r = elm.R1
        vsc.x = elm.X1
        vsc.g0 = elm.G0sw

        vsc.setAllBeq(elm.Beq)
        vsc.beq_max = elm.Beq_max
        vsc.beq_min = elm.Beq_min

        vsc.k = elm.k

        vsc.setAllTapModule(elm.m)
        vsc.tap_max = elm.m_max
        vsc.tap_min = elm.m_min

        vsc.setAllTapPhase(elm.theta)
        vsc.phase_max = elm.theta_max
        vsc.phase_min = elm.theta_min

        vsc.setAllPdcSet(elm.Pdc_set)
        vsc.setAllVacSet(elm.Vac_set)
        vsc.setAllVdcSet(elm.Vdc_set)
        vsc.k_droop = elm.kdp

        vsc.alpha1 = elm.alpha1
        vsc.alpha2 = elm.alpha2
        vsc.alpha3 = elm.alpha3

        vsc.setAllMonitorloading(elm.monitor_loading)
        vsc.setAllContingencyenabled(elm.contingency_enabled)

        if time_series:
            vsc.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            vsc.rates = elm.rate_prof if tidx is None else elm.rate_prof[tidx]
            contingency_rates = elm.rate_prof * elm.contingency_factor
            vsc.contingency_rates = contingency_rates if tidx is None else contingency_rates[tidx]
            vsc.overload_cost = elm.Cost_prof
        else:
            vsc.setAllRates(elm.rate)
            vsc.setAllOverloadCost(elm.Cost)

        npa_circuit.addAcDcConverter(vsc)


def get_dc_line_data(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """
    # Compile the lines
    for i, elm in enumerate(circuit.dc_lines):
        lne = npa.DcLine(uuid=elm.idtag,
                         name=elm.name,
                         calc_node_from=bus_dict[elm.bus_from.idtag],
                         calc_node_to=bus_dict[elm.bus_to.idtag],
                         time_steps=ntime,
                         length=elm.length,
                         rate=elm.rate,
                         active_default=elm.active,
                         r=elm.R,
                         monitor_loading_default=elm.monitor_loading,
                         monitor_contingency_default=elm.contingency_enabled
                         )

        if time_series:
            lne.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            lne.rates = elm.rate_prof if tidx is None else elm.rate_prof[tidx]

            contingency_rates = elm.rate_prof * elm.contingency_factor
            lne.contingency_rates = contingency_rates if tidx is None else contingency_rates[tidx]
            lne.overload_cost = elm.Cost_prof
        else:
            lne.setAllOverloadCost(elm.Cost)

        npa_circuit.addDcLine(lne)


def get_hvdc_data(circuit: MultiCircuit, npa_circuit: "npa.HybridCircuit", bus_dict, time_series: bool, ntime=1, tidx=None):
    """

    :param circuit: GridCal circuit
    :param npa_circuit: Newton circuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param bus_dict: dictionary of bus id to Newton bus object
    :param ntime: number of time steps
    """

    cmode_dict = {HvdcControlType.type_0_free: npa.HvdcControlMode.HvdcControlAngleDroop,
                  HvdcControlType.type_1_Pset: npa.HvdcControlMode.HvdcControlPfix}

    for i, elm in enumerate(circuit.hvdc_lines):
        """
        (uuid: str = '', 
        secondary_id: str = '', 
        name: str = '', 
        calc_node_from: newtonpa.CalculationNode = None, 
        calc_node_to: newtonpa.CalculationNode = None, 
        cn_from: newtonpa.ConnectivityNode = None, 
        cn_to: newtonpa.ConnectivityNode = None, 
        time_steps: int = 1, 
        active_default: int = 1, 
        rate: float = 9999, 
        contingency_rate: float = 9999, 
        monitor_loading_default: int = 1, 
        monitor_contingency_default: int = 1, 
        P: float = 0.0, 
        Vf: float = 1.0, 
        Vf: float = 1.0, 
        r: float = 1e-20, 
        angle_droop: float = 360.0, 
        length: float = 0.0, 
        min_firing_angle_f: float = -1.0, 
        max_firing_angle_f: float = 1.0, 
        min_firing_angle_t: float = -1.0, 
        max_firing_angle_t: float = -1.0, 
        control_mode: newtonpa.HvdcControlMode = <HvdcControlMode.HvdcControlPfix: 1>)
        """
        hvdc = npa.HvdcLine(uuid=elm.idtag,
                            secondary_id=str(elm.code),
                            name=elm.name,
                            calc_node_from=bus_dict[elm.bus_from.idtag],
                            calc_node_to=bus_dict[elm.bus_to.idtag],
                            cn_from=None,
                            cn_to=None,
                            time_steps=ntime,
                            active_default=int(elm.active),
                            rate=elm.rate,
                            contingency_rate=elm.rate * elm.contingency_factor,
                            monitor_loading_default=1,
                            monitor_contingency_default=1,
                            P=elm.Pset,
                            Vf=elm.Vset_f,
                            Vt=elm.Vset_t,
                            r=elm.r,
                            angle_droop=elm.angle_droop,
                            length=elm.length,
                            min_firing_angle_f=elm.min_firing_angle_f,
                            max_firing_angle_f=elm.max_firing_angle_f,
                            min_firing_angle_t=elm.min_firing_angle_t,
                            max_firing_angle_t=elm.max_firing_angle_t,
                            control_mode=cmode_dict[elm.control_mode])

        # hvdc.monitor_loading = elm.monitor_loading
        # hvdc.contingency_enabled = elm.contingency_enabled

        if time_series:
            hvdc.active = elm.active_prof.astype(BINT) if tidx is None else elm.active_prof.astype(BINT)[tidx]
            hvdc.rates = elm.rate_prof if tidx is None else elm.rate_prof[tidx]
            hvdc.Vf = elm.Vset_f_prof if tidx is None else elm.Vset_f_prof[tidx]
            hvdc.Vt = elm.Vset_t_prof if tidx is None else elm.Vset_t_prof[tidx]

            contingency_rates = elm.rate_prof * elm.contingency_factor
            hvdc.contingency_rates = contingency_rates if tidx is None else contingency_rates[tidx]

            hvdc.angle_droop = elm.angle_droop_prof if tidx is None else elm.angle_droop_prof[tidx]
            hvdc.overload_cost = elm.overload_cost_prof
        else:
            hvdc.contingency_rates = elm.rate * elm.contingency_factor
            hvdc.angle_droop = elm.angle_droop
            hvdc.setAllOverloadCost(elm.overload_cost)
            hvdc.setAllControlMode(cmode_dict[elm.control_mode])

        npa_circuit.addHvdcLine(hvdc)


def to_newton_pa(circuit: MultiCircuit, time_series: bool, tidx: List[int] = None, override_branch_controls=False):
    """
    Convert GridCal circuit to Newton
    :param circuit: MultiCircuit
    :param time_series: compile the time series from GridCal? otherwise just the snapshot
    :param tidx: list of time indices
    :param override_branch_controls: If true the branch controls are set to Fix
    :return: npa.HybridCircuit instance
    """

    if tidx is None:
        ntime = circuit.get_time_number() if time_series else 1
        if ntime == 0:
            ntime = 1
    else:
        ntime = len(tidx)

    npaCircuit = npa.HybridCircuit(uuid=circuit.idtag, name=circuit.name, time_steps=ntime)

    area_dict = add_npa_areas(circuit, npaCircuit, ntime)
    zone_dict = add_npa_zones(circuit, npaCircuit, ntime)

    con_groups_dict = add_npa_contingency_groups(circuit, npaCircuit, ntime)
    add_npa_contingencies(circuit, npaCircuit, ntime, con_groups_dict)
    inv_groups_dict = add_npa_investment_groups(circuit, npaCircuit, ntime)
    add_npa_investments(circuit, npaCircuit, ntime, inv_groups_dict)

    bus_dict = add_npa_buses(circuit, npaCircuit, time_series, ntime, tidx, area_dict)
    add_npa_loads(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    add_npa_static_generators(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    add_npa_shunts(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    add_npa_generators(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    get_battery_data(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    add_npa_line(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    get_transformer_data(circuit, npaCircuit, bus_dict, time_series, ntime, tidx, override_branch_controls)
    get_vsc_data(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    get_dc_line_data(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)
    get_hvdc_data(circuit, npaCircuit, bus_dict, time_series, ntime, tidx)

    # npa.FileHandler().save(npaCircuit, circuit.name + "_circuit.newton")

    return npaCircuit, (bus_dict, area_dict, zone_dict)


class FakeAdmittances:

    def __init__(self):
        self.Ybus = None
        self.Yf = None
        self.Yt = None


def get_snapshots_from_newtonpa(circuit: MultiCircuit, override_branch_controls=False):

    from GridCal.Engine.Core.snapshot_pf_data import SnapshotData

    npaCircuit, (bus_dict, area_dict, zone_dict) = to_newton_pa(circuit,
                                                                time_series=False,
                                                                override_branch_controls=override_branch_controls)

    npa_data_lst = npa.compileAt(npaCircuit, t=0).splitIntoIslands()

    data_lst = list()

    for npa_data in npa_data_lst:

        data = SnapshotData(nbus=0,
                            nline=0,
                            ndcline=0,
                            ntr=0,
                            nvsc=0,
                            nupfc=0,
                            nhvdc=0,
                            nload=0,
                            ngen=0,
                            nbatt=0,
                            nshunt=0,
                            nstagen=0,
                            sbase=0,
                            ntime=1)

        conn = npa_data.getConnectivity()
        inj = npa_data.getInjections()
        tpes = npa_data.getSimulationIndices(inj.S0)
        adm = npa_data.getAdmittances(conn)
        lin = npa_data.getLinearMatrices(conn)
        series_adm = npa_data.getSeriesAdmittances(conn)
        fd_adm = npa_data.getFastDecoupledAdmittances(conn, tpes)
        qlim = npa_data.getQLimits()

        data.Vbus_ = npa_data.Vbus.reshape(-1, 1)
        data.Sbus_ = inj.S0.reshape(-1, 1)
        data.Ibus_ = inj.I0.reshape(-1, 1)
        data.branch_data.names = np.array(npa_data.branch_data.names)
        data.branch_data.tap_f = npa_data.branch_data.vtap_f
        data.branch_data.tap_t = npa_data.branch_data.vtap_t

        data.bus_data.names = np.array(npa_data.bus_data.names)

        data.Admittances = FakeAdmittances()
        data.Admittances.Ybus = adm.Ybus
        data.Admittances.Yf = adm.Yf
        data.Admittances.Yt = adm.Yt

        data.Bbus_ = lin.Bbus
        data.Bf_ = lin.Bf

        data.Yseries_ = series_adm.Yseries
        data.Yshunt_ = series_adm.Yshunt

        data.B1_ = fd_adm.B1
        data.B2_ = fd_adm.B2

        data.Cf_ = conn.Cf
        data.Ct_ = conn.Ct

        data.bus_data.bus_types = tpes.types
        data.pq_ = tpes.pq
        data.pv_ = tpes.pv
        data.vd_ = tpes.vd
        data.pqpv_ = tpes.no_slack

        data.original_bus_idx = npa_data.bus_data.original_indices
        data.original_branch_idx = npa_data.branch_data.original_indices

        data.Qmax_bus_ = qlim.qmax_bus
        data.Qmin_bus_ = qlim.qmin_bus

        control_indices = npa_data.getSimulationIndices(Sbus=data.Sbus_)

        data.iPfsh = control_indices.iPfsh
        data.iQfma = control_indices.iQfma
        data.iBeqz = control_indices.iBeqz
        data.iBeqv = control_indices.iBeqv
        data.iVtma = control_indices.iVtma
        data.iQtma = control_indices.iQtma
        data.iPfdp = control_indices.iPfdp
        data.iVscL = control_indices.iVscL
        # data.VfBeqbus = control_indices.iVfBeqBus
        # data.Vtmabus = control_indices.iVtmaBus

        data_lst.append(data)

    return data_lst


def get_newton_pa_pf_options(opt: PowerFlowOptions):
    """
    Translate GridCal power flow options to Newton power flow options
    :param opt:
    :return:
    """
    solver_dict = {SolverType.NR: npa.SolverType.NR,
                   SolverType.DC: npa.SolverType.DC,
                   SolverType.HELM: npa.SolverType.HELM,
                   SolverType.IWAMOTO: npa.SolverType.IWAMOTO,
                   SolverType.LM: npa.SolverType.LM,
                   SolverType.LACPF: npa.SolverType.LACPF,
                   SolverType.FASTDECOUPLED: npa.SolverType.FD
                   }

    q_control_dict = {ReactivePowerControlMode.NoControl: npa.ReactivePowerControlMode.NoControl,
                      ReactivePowerControlMode.Direct: npa.ReactivePowerControlMode.Direct}

    if opt.solver_type in solver_dict.keys():
        solver_type = solver_dict[opt.solver_type]
    else:
        solver_type = npa.SolverType.NR

    """
    solver_type: newtonpa.SolverType = <SolverType.NR: 0>, 
    retry_with_other_methods: bool = True, 
    verbose: bool = False, 
    initialize_with_existing_solution: bool = False, 
    tolerance: float = 1e-06, 
    max_iter: int = 15, 
    control_q_mode: newtonpa.ReactivePowerControlMode = <ReactivePowerControlMode.NoControl: 0>, 
    tap_control_mode: newtonpa.TapsControlMode = <TapsControlMode.NoControl: 0>, 
    distributed_slack: bool = False, 
    ignore_single_node_islands: bool = False, 
    correction_parameter: float = 0.5, 
    mu0: float = 1.0
    """

    return npa.PowerFlowOptions(solver_type=solver_type,
                                retry_with_other_methods=opt.retry_with_other_methods,
                                verbose=opt.verbose,
                                initialize_with_existing_solution=opt.initialize_with_existing_solution,
                                tolerance=opt.tolerance,
                                max_iter=opt.max_iter,
                                control_q_mode=q_control_dict[opt.control_Q],
                                distributed_slack=opt.distributed_slack,
                                correction_parameter=0.5,
                                mu0=opt.mu
                                )


def get_newton_pa_nonlinear_opf_options(pfopt: PowerFlowOptions, opfopt: "OptimalPowerFlowOptions"):
    """
    Translate GridCal power flow options to Newton power flow options
    :param opt:
    :return:
    """
    q_control_dict = {ReactivePowerControlMode.NoControl: npa.ReactivePowerControlMode.NoControl,
                      ReactivePowerControlMode.Direct: npa.ReactivePowerControlMode.Direct}

    """
    tolerance: float = 1e-06, 
    max_iter: int = 20, 
    mu0: float = 1.0, 
    q_control: newtonpa.ReactivePowerControlMode = < ReactivePowerControlMode.Direct: 1 >, 
    flow_control: bool = True, 
    verbose: bool = False
    """

    solver_dict = {bs.MIPSolvers.CBC: npa.LpSolvers.Highs,
                   bs.MIPSolvers.HiGS: npa.LpSolvers.Highs,
                   bs.MIPSolvers.XPRESS: npa.LpSolvers.Xpress,
                   bs.MIPSolvers.CPLEX: npa.LpSolvers.CPLEX,
                   bs.MIPSolvers.GLOP: npa.LpSolvers.Highs,
                   bs.MIPSolvers.SCIP: npa.LpSolvers.Highs,
                   bs.MIPSolvers.GUROBI: npa.LpSolvers.Gurobi}

    return npa.NonlinearOpfOptions(tolerance=pfopt.tolerance,
                                   max_iter=pfopt.max_iter,
                                   mu0=pfopt.mu,
                                   control_q_mode=q_control_dict[pfopt.control_Q],
                                   flow_control=True,
                                   voltage_control=True,
                                   solver=solver_dict[opfopt.mip_solver],
                                   initialize_with_existing_solution=pfopt.initialize_with_existing_solution)


def get_newton_pa_linear_opf_options(opfopt: "OptimalPowerFlowOptions", pfopt: PowerFlowOptions, npa_circuit: "npa.HybridCircuit", area_dict):
    """
    Translate GridCal power flow options to Newton power flow options
    :param opt:
    :return:
    """

    solver_dict = {bs.MIPSolvers.CBC: npa.LpSolvers.Highs,
                   bs.MIPSolvers.HiGS: npa.LpSolvers.Highs,
                   bs.MIPSolvers.XPRESS: npa.LpSolvers.Xpress,
                   bs.MIPSolvers.CPLEX: npa.LpSolvers.CPLEX,
                   bs.MIPSolvers.GLOP: npa.LpSolvers.Highs,
                   bs.MIPSolvers.SCIP: npa.LpSolvers.Highs,
                   bs.MIPSolvers.GUROBI: npa.LpSolvers.Gurobi}

    grouping_dict = {bs.TimeGrouping.NoGrouping: npa.TimeGrouping.NoGrouping,
                     bs.TimeGrouping.Daily: npa.TimeGrouping.Daily,
                     bs.TimeGrouping.Weekly: npa.TimeGrouping.Weekly,
                     bs.TimeGrouping.Monthly: npa.TimeGrouping.Monthly,
                     bs.TimeGrouping.Hourly: npa.TimeGrouping.Hourly}

    from GridCal.Engine.Simulations.OPF.opf_options import OptimalPowerFlowOptions, ZonalGrouping
    opt = npa.LinearOpfOptions()
    opt.solver = solver_dict[opfopt.mip_solver]
    opt.grouping = grouping_dict[opfopt.grouping]
    opt.unit_commitment = False
    opt.compute_flows = opfopt.zonal_grouping == ZonalGrouping.NoGrouping
    opt.check_with_power_flow = False
    opt.add_contingencies = opfopt.consider_contingencies
    opt.skip_generation_limits = opfopt.skip_generation_limits
    opt.maximize_area_exchange = opfopt.maximize_flows
    opt.unit_commitment = opfopt.unit_commitment
    opt.use_ramp_constraints = False
    opt.lodf_threshold = opfopt.lodf_tolerance
    opt.pf_options = get_newton_pa_pf_options(pfopt)

    if opfopt.areas_from is not None:
        opt.areas_from = [area_dict[e] for e in opfopt.areas_from]

    if opfopt.areas_to is not None:
        opt.areas_to = [area_dict[e] for e in opfopt.areas_to]

    return opt


def newton_pa_pf(circuit: MultiCircuit, opt: PowerFlowOptions, time_series=False, tidx=None) -> "npa.PowerFlowResults":
    """
    Newton power flow
    :param circuit: MultiCircuit instance
    :param opt: Power Flow Options
    :param time_series: Compile with GridCal time series?
    :param tidx: Array of time indices
    :param override_branch_controls: If true, the branch controls are set to fix
    :return: Newton Power flow results object
    """
    npa_circuit, (bus_dict, area_dict, zone_dict) = to_newton_pa(circuit,
                                                                 time_series=time_series,
                                                                 tidx=tidx,
                                                                 override_branch_controls=opt.override_branch_controls)

    pf_options = get_newton_pa_pf_options(opt)

    if time_series:
        # it is already sliced to the relevant time indices
        time_indices = [i for i in range(circuit.get_time_number())]
        n_threads = 0  # max threads
    else:
        time_indices = [0]
        n_threads = 1

    pf_res = npa.runPowerFlow(circuit=npa_circuit,
                              pf_options=pf_options,
                              time_indices=time_indices,
                              n_threads=n_threads,
                              V0=circuit.get_voltage_guess() if opt.initialize_with_existing_solution else None)

    return pf_res


def newton_pa_linear_opf(circuit: MultiCircuit, opf_options, pfopt: PowerFlowOptions,
                         time_series=False, tidx=None) -> "npa.LinearOpfResults":
    """
    Newton power flow
    :param circuit: MultiCircuit instance
    :param pfopt: Power Flow Options
    :param time_series: Compile with GridCal time series?
    :param tidx: Array of time indices
    :return: Newton Power flow results object
    """
    npaCircuit, (bus_dict, area_dict, zone_dict) = to_newton_pa(circuit=circuit,
                                                                time_series=time_series,
                                                                tidx=tidx,
                                                                override_branch_controls=False)

    if time_series:
        # it is already sliced to the relevant time indices
        time_indices = [i for i in range(circuit.get_time_number())]
        n_threads = 0  # max threads
    else:
        time_indices = [0]
        n_threads = 1

    options = get_newton_pa_linear_opf_options(opf_options, pfopt, npaCircuit, area_dict)

    pf_res = npa.runLinearOpf(circuit=npaCircuit,
                              options=options,
                              time_indices=time_indices,
                              n_threads=n_threads,
                              mute_pg_bar=False)

    return pf_res


def newton_pa_nonlinear_opf(circuit: MultiCircuit, pfopt: PowerFlowOptions, opfopt: "OptimalPowerFlowOptions",
                            time_series=False, tidx=None) -> "npa.NonlinearOpfResults":
    """
    Newton power flow
    :param circuit: MultiCircuit instance
    :param pfopt: Power Flow Options
    :param time_series: Compile with GridCal time series?
    :param tidx: Array of time indices
    :return: Newton Power flow results object
    """
    npaCircuit, (bus_dict, area_dict, zone_dict) = to_newton_pa(circuit=circuit,
                                                                time_series=time_series,
                                                                tidx=tidx,
                                                                override_branch_controls=False)

    pf_options = get_newton_pa_nonlinear_opf_options(pfopt, opfopt)

    if time_series:
        # it is already sliced to the relevant time indices
        time_indices = [i for i in range(circuit.get_time_number())]
        n_threads = 0  # max threads
    else:
        time_indices = [0]
        n_threads = 1

    pf_res = npa.runNonlinearOpf(circuit=npaCircuit,
                                 pf_options=pf_options,
                                 time_indices=time_indices,
                                 n_threads=n_threads,
                                 mute_pg_bar=False,
                                 V0=circuit.get_voltage_guess() if pfopt.initialize_with_existing_solution else None)

    return pf_res


def newton_pa_linear_matrices(circuit: MultiCircuit, distributed_slack=False, override_branch_controls=False):
    """
    Newton linear analysis
    :param circuit: MultiCircuit instance
    :param distributed_slack: distribute the PTDF slack
    :return: Newton LinearAnalysisMatrices object
    """
    npa_circuit, (bus_dict, area_dict, zone_dict) = to_newton_pa(circuit=circuit,
                                                                 time_series=False,
                                                                 override_branch_controls=override_branch_controls)

    options = npa.LinearAnalysisOptions(distribute_slack=distributed_slack)
    results = npa.runLinearAnalysisAt(t=0, circuit=npa_circuit, options=options)

    return results


def convert_bus_types(arr: List["npa.BusType"]):

    tpe = np.zeros(len(arr), dtype=int)
    for i, val in enumerate(arr):
        if val == npa.BusType.VD:
            tpe[i] = 3
        elif val == npa.BusType.PV:
            tpe[i] = 2
        elif val == npa.BusType.PQ:
            tpe[i] = 1
    return tpe


def translate_newton_pa_pf_results(grid: "MultiCircuit", res: "npa.PowerFlowResults") -> "PowerFlowResults":
    results = PowerFlowResults(n=grid.get_bus_number(),
                               m=grid.get_branch_number_wo_hvdc(),
                               n_tr=grid.get_transformers2w_number(),
                               n_hvdc=grid.get_hvdc_number(),
                               bus_names=res.bus_names,
                               branch_names=res.branch_names,
                               transformer_names=[],
                               hvdc_names=res.hvdc_names,
                               bus_types=res.bus_types)

    results.voltage = res.voltage[0, :]
    results.Sbus = res.Scalc[0, :]
    results.Sf = res.Sf[0, :]
    results.St = res.St[0, :]
    results.loading = res.Loading[0, :]
    results.losses = res.Losses[0, :]
    # results.Vbranch = res.Vbranch[0, :]
    # results.If = res.If[0, :]
    # results.It = res.It[0, :]
    results.Beq = res.Beq[0, :]
    results.m = res.tap_module[0, :]
    results.theta = res.tap_angle[0, :]
    results.F = res.F
    results.T = res.T
    results.hvdc_F = res.hvdc_F
    results.hvdc_T = res.hvdc_T
    results.hvdc_Pf = res.hvdc_Pf[0, :]
    results.hvdc_Pt = res.hvdc_Pt[0, :]
    results.hvdc_loading = res.hvdc_loading[0, :]
    results.hvdc_losses = res.hvdc_losses[0, :]
    results.bus_area_indices = grid.get_bus_area_indices()
    results.area_names = [a.name for a in grid.areas]
    results.bus_types = convert_bus_types(res.bus_types[0])  # this is a list of lists

    for rep in res.stats[0]:
        report = bs.ConvergenceReport()
        for i in range(len(rep.converged)):
            report.add(method=rep.solver[i].name,
                       converged=rep.converged[i],
                       error=rep.norm_f[i],
                       elapsed=rep.elapsed[i],
                       iterations=rep.iterations[i])
            results.convergence_reports.append(report)

    return results


def translate_newton_pa_opf_results(res: "npa.NonlinearOpfResults") -> "OptimalPowerFlowResults":

    from GridCal.Engine.Simulations.OPF.opf_results import OptimalPowerFlowResults
    results = OptimalPowerFlowResults(bus_names=res.bus_names,
                                      branch_names=res.branch_names,
                                      load_names=res.load_names,
                                      generator_names=res.generator_names,
                                      battery_names=res.battery_names,
                                      Sbus=res.Scalc[0, :],
                                      voltage=res.voltage[0, :],
                                      load_shedding=res.load_shedding[0, :],
                                      hvdc_names=res.hvdc_names,
                                      hvdc_power=res.hvdc_Pf[0, :],
                                      hvdc_loading=res.hvdc_loading[0, :],
                                      phase_shift=res.tap_angle[0, :],
                                      bus_shadow_prices=res.bus_shadow_prices[0, :],
                                      generator_shedding=res.generator_shedding[0, :],
                                      battery_power=res.battery_p[0, :],
                                      controlled_generation_power=res.generator_p[0, :],
                                      Sf=res.Sf[0, :],
                                      St=res.St[0, :],
                                      overloads=res.branch_overload[0, :],
                                      loading=res.Loading[0, :],
                                      rates=res.rates[0, :],
                                      contingency_rates=res.contingency_rates[0, :],
                                      converged=res.converged[0],
                                      bus_types=convert_bus_types(res.bus_types[0]))

    results.contingency_flows_list = list()
    results.losses = res.Losses[0, :]

    return results


def debug_newton_pa_circuit_at(npa_circuit: "npa.HybridCircuit", t: int = None):

    if t is None:
        t = 0

    data = npa.compileAt(npa_circuit, t=t)

    for i in range(len(data)):

        print('_' * 200)
        print('Island', i)
        print('_' * 200)

        print("Ybus")
        print(data[i].admittances.Ybus.toarray())

        print('Yseries')
        print(data[i].split_admittances.Yseries.toarray())

        print('Yshunt')
        print(data[i].split_admittances.Yshunt)

        print("Bbus")
        print(data[i].linear_admittances.Bbus.toarray())

        print('B1')
        print(data[i].fast_decoupled_admittances.B1.toarray())

        print('B2')
        print(data[i].fast_decoupled_admittances.B2.toarray())

        print('Sbus')
        print(data[i].Sbus)

        print('Vbus')
        print(data[i].Vbus)

        print('Qmin')
        print(data[i].Qmin_bus)

        print('Qmax')
        print(data[i].Qmax_bus)


if __name__ == '__main__':

    from GridCal.Engine import *

    # fname = '/home/santi/Documentos/Git/GitHub/GridCal/Grids_and_profiles/grids/IEEE14_from_raw.gridcal'
    fname = '/home/santi/Documentos/Git/GitHub/GridCal/Grids_and_profiles/grids/IEEE39.gridcal'
    _grid = FileOpen(fname).open()

    # _newton_grid = to_newton_pa(circuit=_grid, time_series=False)
    _options = PowerFlowOptions()
    _res = newton_pa_pf(circuit=_grid, opt=_options, time_series=True)

    _res2 = translate_newton_pa_pf_results(_grid, _res)

    print()
