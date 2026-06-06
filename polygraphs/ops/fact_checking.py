"""
Ops that contain fact-checkers
"""

import torch

from .complex import UnreliableOp, UnreliableNetworkBasicGullibleBinomialOp, UnreliableNetworkBasicGullibleNegativeEpsOp


class BaseFactCheckersOp(UnreliableOp):
    fact_checker = 0
    unreliable = 1
    user = 2

    def __init__(self, graph, params, silent=True):
        super().__init__(graph, params)

        self.current_step = 0
        self.fact_checkers_activated = False

        self.group_labels = {
            self.fact_checker: "Fact-checkers",
            self.unreliable: "Unreliable",
            self.user: "Users"
        }

        fc_ratio = params.fact_checker_ratio
        reliable = params.reliability

        weights = torch.tensor([
            reliable * fc_ratio,
            1 - reliable,
            reliable * (1 - fc_ratio),
        ], dtype=torch.float)

        group_assignments = torch.multinomial(weights, graph.num_nodes(), replacement=True)
        graph.ndata['group'] = group_assignments.to(self._device)

        print("Fact-checker ID:", self.fact_checker)
        print("Group counts:", {self.group_labels[i]: (group_assignments == i).sum().item() for i in range(3)})

        reliability = torch.ones(graph.num_nodes(), dtype=torch.float)
        reliability[group_assignments == self.unreliable] = 0
        graph.ndata["reliability"] = reliability.to(params.device)

    def _assign_by_degree(self, graph, params, central_group, group_assignments):
        n = graph.num_nodes()
        n_fc = (group_assignments == self.fact_checker).sum().item()
        n_unreliable = (group_assignments == self.unreliable).sum().item()

        degrees = graph.in_degrees().float()
        sorted_nodes = torch.argsort(degrees, descending=True)

        group_assignments = torch.zeros(n, dtype=torch.long)

        if central_group == self.fact_checker:
            top_nodes = sorted_nodes[:n_fc]
            remaining = sorted_nodes[n_fc:]
            group_assignments[top_nodes] = self.fact_checker
            perm = remaining[torch.randperm(len(remaining))]
            group_assignments[perm[:n_unreliable]] = self.unreliable
            group_assignments[perm[n_unreliable:]] = self.user

        elif central_group == self.unreliable:
            top_nodes = sorted_nodes[:n_unreliable]
            remaining = sorted_nodes[n_unreliable:]
            group_assignments[top_nodes] = self.unreliable
            perm = remaining[torch.randperm(len(remaining))]
            group_assignments[perm[:n_fc]] = self.fact_checker
            group_assignments[perm[n_fc:]] = self.user
            
        print("Top 5 degrees:", degrees[sorted_nodes[:5]].tolist())
        print("Central group node degrees:", degrees[group_assignments == central_group].tolist())

        return group_assignments

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


class FactCheckersCentralOp(BaseFactCheckersOp):
    """Fact-checkers are assigned to the highest-degree nodes."""

    def __init__(self, graph, params, silent=True):
        super().__init__(graph, params, silent)

        group_assignments = self._assign_by_degree(
            graph, params,
            central_group=self.fact_checker,
            group_assignments=graph.ndata['group'],
        )
        graph.ndata['group'] = group_assignments.to(self._device)

        print("Group counts (degree-central fact-checkers):",
              {self.group_labels[i]: (group_assignments == i).sum().item() for i in range(3)})

        reliability = torch.ones(graph.num_nodes(), dtype=torch.float)
        reliability[group_assignments == self.unreliable] = 0
        graph.ndata["reliability"] = reliability.to(params.device)


class UnreliableCentralOp(BaseFactCheckersOp):
    """Unreliable nodes are assigned to the highest-degree nodes."""

    def __init__(self, graph, params, silent=True):
        super().__init__(graph, params, silent)

        group_assignments = self._assign_by_degree(
            graph, params,
            central_group=self.unreliable,
            group_assignments=graph.ndata['group'],
        )
        graph.ndata['group'] = group_assignments.to(self._device)

        print("Group counts (degree-central unreliable):",
              {self.group_labels[i]: (group_assignments == i).sum().item() for i in range(3)})

        reliability = torch.ones(graph.num_nodes(), dtype=torch.float)
        reliability[group_assignments == self.unreliable] = 0
        graph.ndata["reliability"] = reliability.to(params.device)


class FactCheckersGulBinOp(BaseFactCheckersOp, UnreliableNetworkBasicGullibleBinomialOp):
    pass


class FactCheckersGulNegEpsOp(UnreliableNetworkBasicGullibleNegativeEpsOp, BaseFactCheckersOp):
    pass


class FactCheckersCentralGulBinOp(FactCheckersCentralOp, UnreliableNetworkBasicGullibleBinomialOp):
    pass


class UnreliableCentralGulBinOp(UnreliableCentralOp, UnreliableNetworkBasicGullibleBinomialOp):
    pass

class UnreliableCentralGulNegEpsOp(UnreliableCentralOp, UnreliableNetworkBasicGullibleNegativeEpsOp):
    pass

class FactCheckersCentralGulNegEpsOp(FactCheckersCentralOp, UnreliableNetworkBasicGullibleNegativeEpsOp):
    pass
