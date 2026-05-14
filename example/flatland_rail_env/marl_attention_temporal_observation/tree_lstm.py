"""
TreeLSTM – Deterministischer Baum-Aggregator für lokale Suchdaten
=================================================================

Stellt eine leichtgewichtige, numpy-basierte Aggregation von baumstrukturierten
lokalen Suchdaten bereit.  Wird in DecisionPointObservation verwendet, um einen
aggregierten Risikoscore aus der tiefenbegrenzten Pfadexploration zu berechnen.

Dies ist KEIN lernbares neuronales Netzwerk – es verwendet feste exponentielle
Tiefengewichtung, um folgende vier Knotenmerkmale zu aggregieren:

    1. deadlock_risk   – Deadlock-Risiko des Knotens [0, 1]
    2. branching       – Normierte Anzahl ausgehender Transitionen [0, 1]
    3. conflict        – Anzahl begegneter Fremd-Agenten (normiert) [0, 1]
    4. depth_norm      – Tiefennähe zum Startknoten (1 / (1 + depth)) [0, 1]

Der resultierende Vektor der Länge ``hidden_dim`` wird zyklisch aus diesen
4 Basiswerten aufgebaut und dient als kompakter Kontext-Eingang in
DecisionPointObservation (Feature-Slot [64]).

Schnittstelle
-------------
>>> lstm = TreeLSTM(input_dim=8, hidden_dim=16)
>>> out = lstm.aggregate(tree_data)   # np.ndarray, shape (hidden_dim,)
"""

import numpy as np


class TreeLSTM:
    """Deterministischer Baum-Aggregator mit tiefengewichteter Abnahme.

    Parameters
    ----------
    input_dim : int
        Dimensionalität der Eingangsfeatures pro Knoten (wird als Metadaten
        gespeichert; die eigentliche Aggregation verwendet 4 interne Features).
    hidden_dim : int
        Dimensionalität des Ausgabevektors.
    depth_decay : float
        Exponentieller Abnahmefaktor pro Tiefenstufe (< 1 → weiter entfernte
        Knoten tragen weniger bei).  Standard: 0.7.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        depth_decay: float = 0.7,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth_decay = float(depth_decay)

    # ------------------------------------------------------------------
    def aggregate(self, tree_data: list) -> np.ndarray:
        """Aggregiert Baumknoten-Features zu einem einzigen Vektor.

        Parameters
        ----------
        tree_data : list[dict]
            Liste von Knoten-Dicts aus ``DecisionPointObservation._local_search``.
            Jedes Dict enthält mindestens:
              - ``depth``              (int)   Tiefe ab Wurzelknoten
              - ``deadlock_risk``      (float) Deadlock-Risikoscore [0, 1]
              - ``num_transitions``    (int)   Anzahl ausgehender Transitionen
              - ``agents_encountered`` (list)  Handles anderer Agenten am Knoten

        Returns
        -------
        np.ndarray
            Aggregierter Vektor der Form ``(hidden_dim,)``, Werte in [0, 1].
        """
        if not tree_data:
            return np.zeros(self.hidden_dim, dtype=np.float32)

        total_weight = 0.0
        agg = np.zeros(4, dtype=np.float64)  # [deadlock, branching, conflict, depth_norm]

        for node in tree_data:
            depth = int(node.get("depth", 0))
            weight = self.depth_decay ** depth

            deadlock_risk = float(node.get("deadlock_risk", 0.0))
            branching = float(min(1.0, node.get("num_transitions", 1) / 3.0))
            conflict = float(min(1.0, len(node.get("agents_encountered", [])) / 2.0))
            depth_norm = 1.0 / (1.0 + depth)

            agg[0] += weight * deadlock_risk
            agg[1] += weight * branching
            agg[2] += weight * conflict
            agg[3] += weight * depth_norm
            total_weight += weight

        if total_weight > 0.0:
            agg /= total_weight

        # Projiziere 4D-Aggregat zyklisch auf hidden_dim
        out = np.empty(self.hidden_dim, dtype=np.float32)
        for i in range(self.hidden_dim):
            out[i] = float(np.clip(agg[i % 4], 0.0, 1.0))

        return out

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TreeLSTM(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"depth_decay={self.depth_decay})"
        )
