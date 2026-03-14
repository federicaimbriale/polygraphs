"""
Ops that contain fact-checkers
"""

import abc
import torch

from .complex import UnreliableOp,  UnreliableNetworkBasicGullibleBinomialOp, UnreliableNetworkBasicGullibleNegativeEpsOp
  
class BaseFactCheckersOp(UnreliableOp):
    fact_checker = 0
    unreliable = 1
    user = 2

    def __init__(self, graph, params, silent=True):
        super().__init__(graph, params, silent=silent)

        self.current_step = 0
        self.fact_checkers_activated = False

        self.group_labels = {
            self.fact_checker: "Fact-checkers",
            self.unreliable: "Unreliable",
            self.user: "Users"
        }

        weights = torch.tensor([0.10, 1 - (params.reliability), 0.65], dtype=torch.float)
  

        group_assignments = torch.multinomial(weights, graph.num_nodes(), replacement=True)
        graph.ndata['group'] = group_assignments.to(self._device)

        print("Fact-checker ID:", self.fact_checker)
        print("Group counts:", {self.group_labels[i]: (group_assignments == i).sum().item() for i in range(3)})

        reliability = torch.ones(graph.num_nodes(), dtype=torch.float)
        reliability[group_assignments == self.unreliable] = 0
        graph.ndata["reliability"] = reliability.to(params.device)

    def set_current_step(self, step):
        self.current_step = step

    def block(self, graph, params):
        if self.current_step < params.simulation.block:
            mask = graph.ndata['group'] == self.fact_checker
            graph.ndata['beliefs'] = graph.ndata['beliefs'] * ~mask
            graph.edata['blocked'] = mask[graph.edges()[0]] | mask[graph.edges()[1]]
        elif not self.fact_checkers_activated:
            mask = graph.ndata['group'] == self.fact_checker
            graph.ndata['beliefs'][mask] = 0.99
            print(f"[Step {self.current_step}] Activated {mask.sum().item()} fact-checkers: beliefs set to 0.99")
            self.fact_checkers_activated = True

    def filterfn(self):
        def function(edges):
            src_group = edges.src['group']
            dst_group = edges.dst['group']

            fact_checker_to_unreliable = (
                ((src_group == self.fact_checker) & (dst_group == self.unreliable)) |
                ((src_group == self.unreliable) & (dst_group == self.fact_checker))
            )
            unreliable_to_user = (
                ((src_group == self.unreliable) & (dst_group == self.user)) |
                ((src_group == self.user) & (dst_group == self.unreliable))
            )

            unreliable_nodes_connected = (
                fact_checker_to_unreliable[edges.src['group']] |
                fact_checker_to_unreliable[edges.dst['group']]
            )

            block_edge = unreliable_to_user & unreliable_nodes_connected
            return ~block_edge

        return function

class FactCheckersGulBinOp(BaseFactCheckersOp, UnreliableNetworkBasicGullibleBinomialOp):
    pass

class FactCheckersGulNegEpsOp(UnreliableNetworkBasicGullibleNegativeEpsOp, BaseFactCheckersOp):
    pass
