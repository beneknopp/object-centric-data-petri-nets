import pm4py
from pm4py.algo.conformance.alignments.petri_net.algorithm import apply_log as alignments_apply_log
from event_log_processing.event_log_manager import EventLogManager, EventLogParameters
from rule_mining.decision_mining import DecisionMiner
from rule_mining.interaction_mining import InteractionMiner
from alignment_processing.utils import CaseAlignment
import pickle


initial_steps = True and False
alignment_steps = True and False
must_mine_decisions = True# and False
must_mine_interactions = True and False
must_visualize_ocdpn = True# and False

ocel2_sqlite_path = "sample_logs/order-management.sqlite"
event_log_manager_name = "elmo"
if initial_steps:
    elmo = EventLogManager(event_log_manager_name)
    elmo_parameters = {
        EventLogParameters.ALLOWED_OBJECT_TYPES: ["orders", "items", "employees"],
        EventLogParameters.ALLOWED_ACTIVITIES: ["place order", "confirm order", "item out of stock", "reorder item", "pick item"]
    }
    elmo.load_ocel2_sqlite(ocel2_sqlite_path, elmo_parameters)
    elmo.create_dataframes()
    pm4py_ocel = elmo.ocel2
    pm4py_ocpn = pm4py.discover_oc_petri_net(pm4py_ocel)
    pm4py.save_vis_ocpn(pm4py_ocpn, "output/ocpn.svg")
    with open("output/p4mpy_ocpn.pkl", "wb") as file:
        pickle.dump(pm4py_ocpn, file)
    elmo.save()
else:
    pm4py_ocel = pm4py.read_ocel2_sqlite("sample_logs/order-management.sqlite")
    file = open("output/p4mpy_ocpn.pkl", "rb")
    pm4py_ocpn = pickle.load(file)
    elmo = EventLogManager.load(event_log_manager_name)

decision_miner_name = "demi"

if alignment_steps:
    objects = elmo.get_objects_frame()
    object_types = elmo.get_object_types()
    object_types = sorted(object_types, key=lambda x: -len(objects[objects["ocel:type"] == x]))
    petri_nets = {}
    flat_logs = {}
    alignments = {}
    trace_idx = {}
    for object_type in object_types:
        pn_name = "pn_" + object_type
        petri_net = pm4py_ocpn["petri_nets"][object_type]
        pn, im, fm = petri_net
        places = pn.places
        transitions = pn.transitions
        arcs = pn.arcs
        arc_sources = [arc.source for arc in arcs]
        arc_targets = [arc.target for arc in arcs]
        petri_nets[object_type] = pn
        flat_log = pm4py.ocel_flattening(pm4py_ocel, object_type)
        flat_logs[object_type] = flat_log
        alignments_ot = alignments_apply_log(flat_log, pn, im, fm)
        case_ids_ot = list(flat_log.groupby("case:concept:name").groups.keys())
        case_alignments_ot = list(map(lambda idx: CaseAlignment(case_ids_ot[idx], alignments_ot[idx]["alignment"]), range(len(case_ids_ot))))
        alignments[object_type] = case_alignments_ot
    #alignments =\
    #    calculate_oc_alignments(working_ocel, working_ocpn)
    with open("alignment_processing/alignments.pkl", "wb") as wf:
        pickle.dump(alignments, wf)
    with open("output/petri_nets.pkl", "wb") as wf:
        pickle.dump(petri_nets, wf)
else:
    with open("alignment_processing/alignments.pkl", "rb") as rf:
        alignments = pickle.load(rf)
    with open("output/petri_nets.pkl", "rb") as rf:
        petri_nets = pickle.load(rf)


if must_mine_decisions:
    event_object_attributes_frame = elmo.get_event_object_attributes_frame()
    o2o_frame = elmo.get_o2o_frame()
    object_types = elmo.get_object_types()
    demi = DecisionMiner(decision_miner_name, object_types, petri_nets, alignments, elmo)
    demi.mine_decision_trees()
    demi.save()
else:
    demi = DecisionMiner.load(decision_miner_name)


pattern_miner_name = "pamela"
activities = elmo.get_activities()
object_types = elmo.get_object_types()
if must_mine_interactions:
    pamela = InteractionMiner(pattern_miner_name, activities, object_types, pm4py_ocpn, elmo)
    pamela.run()
    pamela.save()
else:
    pamela = InteractionMiner.load(pattern_miner_name)

if must_visualize_ocdpn:
    activities = elmo.get_activities()
    pamela.transform_patterns_to_conditions()
    pm4py_ocel = elmo.ocel2
    with open("output/p4mpy_ocpn.pkl", "rb") as file:
        pm4py_ocpn = pickle.load(file)
    guards = {}
    for act in activities:
        guard = pamela.get_interaction_condition(act).print_sml()
        guards[act] = guard
    pm4py.save_vis_ocpn(pm4py_ocpn, "output/ocpn.svg", guards = guards)

    for object_type in object_types:
        petri_net = pm4py_ocpn["petri_nets"][object_type]
        pn, im, fm = petri_net
        transitions = pn.transitions
        decorations = {}
        for tr in transitions:
            label = tr.label
            transition_name = tr.name
            guard = demi.get_decision_rule(object_type, transition_name)
            if guard is not None:
                label = label if label is not None else transition_name
                label = label + ":\n" + guard.print_sml()
            decorations[tr] = {"label": label if label is not None else "", "color": "white"}
        pn_name = "pn_" + object_type
        pm4py.save_vis_petri_net(pn, im, fm, "output/" + pn_name + ".svg", decorations=decorations)