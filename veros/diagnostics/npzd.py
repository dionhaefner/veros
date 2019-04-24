import logging

from .diagnostic import VerosDiagnostic
from .. import veros_class_method
from graphviz import Digraph
import numpy as np

class NPZDMonitor(VerosDiagnostic):
    """Diagnostic monitoring nutrients and plankton concentrations
    """

    name = "npzd"
    output_frequency = None
    restart_attributes = []
    output_variables = []

    def __init__(self, setup):
            self.save_graph = False
            self.npzd_graph = Digraph("npzd_dynamics", filename="npzd_dynamics.gv")
            self.npzd_graph.graph_attr["splines"] = "ortho"
            self.npzd_graph.graph_attr["nodesep"] = "1"
            self.npzd_graph.graph_attr["node"] = "square"

    def initialize(self, vs):
        pass


    def diagnose(self, vs):
        pass


    @veros_class_method
    def output(self, vs):
        """Print NPZD interaction graph
        """
        for tracer in vs.npzd_tracers:
            self.npzd_graph.node(tracer)

        for rule, source, sink, label in vs.npzd_rules:
            self.npzd_graph.edge(source, sink, xlabel=label)

        for rule, source, sink, label in vs.npzd_pre_rules:
            self.npzd_graph.edge(source, sink, xlabel=label, style="dotted")

        for rule, source, sink, label in vs.npzd_post_rules:
            self.npzd_graph.edge(source, sink, xlabel=label, style="dashed")

        if vs.sinking_speeds:
            self.npzd_graph.node("Bottom", shape="square")
            for sinker in vs.sinking_speeds:
                self.npzd_graph.edge(sinker, "Bottom", xlabel="sinking")

        if self.save_graph:
            self.save_graph = False
            self.npzd_graph.save()



    def read_restart(self, vs):
        pass

    def write_restart(self, vs, outfile):
        pass