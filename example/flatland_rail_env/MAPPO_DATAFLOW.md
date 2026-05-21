# MAPPO Policy — Datenfluss & Funktionale Notation

Dokumentation des Forward-Pass für einen einzelnen Agenten `i` im
`MARL_ATTENTION_TEMPORAL_PPOPolicy` (Datei: `marl_attention_temporal_mappo.py`).

---

## 1. Top-Level

$$
\pi(\cdot \mid s_i),\; V(s_i) \;=\; \text{Heads}\Big(\text{Policy}\big(\text{Wrap}(\text{Env}(i))\big)\Big)
$$

---

## 2. Environment liefert pro Agent

$$
\text{Env}(i) \;=\; \big(\,b_i,\; S_i,\; T_i\,\big)
$$

mit
- $b_i \in \mathbb{R}^{13}$ — Base-Features (path_*, delta_*, st_3, priority_rank, is_pre_merge, is_switch, sp_*)
- $S_i \subseteq \{1,\dots,N\}\setminus\{i\}$ — Set sichtbarer Nachbarn (`seen_agents` aus `local_search` depth=4)
- $T_i$ — Tree-Payload `{nodes, edges}`

---

## 3. Wrapper sammelt zentral + erweitert um Nachbar-Obs

$$
\text{Wrap}(i) \;=\; \Big(\, b_i,\; \{\,b_j\,\}_{j \in S_i^\top},\; T_i \,\Big)
$$

mit $S_i^\top \subseteq S_i$, $|S_i^\top| \le 3$ (Top-3 nach Relevance-Score).

Über 3 Zeitschritte (temporal window):

$$
\text{seq}_i \;=\; \big[\,\text{Wrap}_{t-2}^{(i)},\; \text{Wrap}_{t-1}^{(i)},\; \text{Wrap}_t^{(i)}\,\big]
$$

---

## 4. Policy-Netz als Komposition

$$
\text{Policy}(\text{seq}_i) \;=\; (\Phi_4 \circ \Phi_3 \circ \Phi_2 \circ \Phi_1)(\text{seq}_i)
$$

### 4.1 $\Phi_1$ — Temporal Self-Attention (nur Self)

$$
\Phi_1\big(\text{seq}_i\big) \;=\; \text{Attn}_\text{temp}\Big(\big[\,\phi_\text{obs}(b_i^{t-2}),\; \phi_\text{obs}(b_i^{t-1}),\; \phi_\text{obs}(b_i^{t})\,\big] + \text{PE}\Big)_{[-1]}
\;=\; h_i^{\text{temp}} \in \mathbb{R}^{128}
$$

mit $\phi_\text{obs}: \mathbb{R}^{13} \to \mathbb{R}^{128}$ (shared `obs_encoder`).

### 4.2 $\Phi_2$ — Tree-Fusion (additiv, nur Self)

$$
\Phi_2\big(h_i^{\text{temp}}, T_i^t\big) \;=\; \text{LN}\Big(\, h_i^{\text{temp}} \;+\; \phi_\text{tree}(T_i^t)\,\Big) \;=\; h_i^{\text{self}}
$$

mit $\phi_\text{tree}$ = `tree_payload_encoder` (verarbeitet Knoten + Kanten zu einem 128-dim Vektor).

### 4.3 $\Phi_3$ — Spatial Cross-Attention (Self ↔ Opponents)

Opponent-Embeddings über denselben shared encoder:

$$
e_j \;=\; W_\text{proj}\big(\phi_\text{obs}(b_j^t)\big),\quad j \in S_i^\top
$$

$$
e_i^\text{self} \;=\; W_\text{proj}(h_i^{\text{self}})
$$

Cross-Attention mit Self als Query:

$$
\Phi_3 \;=\; \text{Attn}_\text{spat}\Big(Q=e_i^\text{self},\; K=V=[\,e_i^\text{self},\, e_{j_1},\, \dots,\, e_{j_m}\,]\Big) \;+\; h_i^{\text{self}}
\;=\; c_i \in \mathbb{R}^{128}
$$

### 4.4 $\Phi_4$ — Communication (gerichtete Messages mit Gate + Intent)

Pro Sender $j \in S_i^\top$:

$$
\begin{aligned}
m_j &= \tanh\!\big(W_\text{msg}\,e_j \;+\; \text{softmax}(W_\text{int}\,e_j) \cdot E_\text{intent}\big)  &\text{(Message + Intent)}\\
g_j &= \sigma\!\big(w_\text{gate}^\top e_j\big)  &\text{(Sender-Gate)}\\
\alpha_j &= \frac{\exp\!\big(\langle W_K e_j,\, W_Q c_i\rangle / \sqrt{H}\big)}{\sum_{j'} \exp(\cdot)} &\text{(Receiver-Addressing)}\\
w_j &= \frac{\alpha_j \cdot g_j}{\sum_{j'} \alpha_{j'} g_{j'} + \varepsilon}
\end{aligned}
$$

mit $E_\text{intent} \in \mathbb{R}^{3 \times 128}$ = `intent_embedding` (Slots WAIT/GO/YIELD).

Aggregation:

$$
\text{comm}_i \;=\; \sum_{j \in S_i^\top} w_j \cdot m_j
$$

$$
\Phi_4 \;=\; \text{LN}\big(c_i + \text{comm}_i\big) \;=\; z_i \in \mathbb{R}^{128}
$$

Final-Projektion:

$$
z_i^\star \;=\; W_\text{out}\,z_i
$$

---

## 5. Heads

$$
\begin{aligned}
\text{logits}_i &= W_\text{actor}\,z_i^\star \in \mathbb{R}^{5}\\
\tilde{\ell}_i &= \text{logits}_i + \beta_\text{SP}\cdot \mathbb{1}_{a = \text{SP}(s_i)} + \mathcal{M}_i \quad (\mathcal{M}_i = -\infty \text{ für maskierte Aktionen})\\
\pi(a \mid s_i) &= \text{softmax}(\tilde{\ell}_i)\\
a_i &\sim \pi(\cdot \mid s_i)\\[6pt]
V(s_i) &= W_\text{critic}\,z_i^\star \in \mathbb{R}
\end{aligned}
$$

---

## 6. Komplette Pipeline in einer Zeile

$$
\pi_i,\; V_i \;=\; \text{Heads}\circ\, W_\text{out}\,\circ\,
\underbrace{\text{LN}\big(\,\cdot + \text{Comm}(\cdot,\{e_j\})\,\big)}_{\Phi_4}\,\circ\,
\underbrace{\text{Attn}_\text{spat}(\,\cdot, \{e_j\})}_{\Phi_3}\,\circ\,
\underbrace{\text{LN}\big(\,\cdot + \phi_\text{tree}(T_i)\big)}_{\Phi_2}\,\circ\,
\underbrace{\text{Attn}_\text{temp}\!\big(\phi_\text{obs}(b_i^{t-2:t})\big)}_{\Phi_1}
$$

mit

$$
e_j = W_\text{proj}\,\phi_\text{obs}(b_j^t),\quad j \in S_i^\top \;=\; \text{TopK}_3\big(\text{relevance}(j)\big)
$$

---

## 7. Dimensions-Tracking

| Symbol | Bedeutung | Dimension |
|---|---|---|
| $b_i$ | base features | $\mathbb{R}^{13}$ |
| $T_i$ | tree payload | variabel (≤8 Knoten, ≤10 Kanten) |
| $S_i^\top$ | Top-K Nachbarn | $\le 3$ |
| $\phi_\text{obs}(b)$ | obs encoder output | $\mathbb{R}^{128}$ |
| $\phi_\text{tree}(T)$ | tree encoder output | $\mathbb{R}^{128}$ |
| $h_i^{\text{temp}}, h_i^{\text{self}}, c_i, z_i, z_i^\star$ | Zwischen-States | $\mathbb{R}^{128}$ |
| $m_j$ | message vector | $\mathbb{R}^{128}$ |
| $g_j, \alpha_j, w_j$ | Gate / Attention / Final-Weight | $\mathbb{R}$ |
| $\pi$ | Action-Verteilung | $\Delta^5$ (Simplex über 5 Aktionen) |
| $V$ | State-Value | $\mathbb{R}$ |

---

## 8. Parameter-Sharing Hinweis

- $\phi_\text{obs}$ — shared zwischen Self und allen Opponents (eine Instanz)
- $\phi_\text{tree}$ — nur auf Self angewendet
- Actor- und Critic-Encoder — getrennte Parameter (`encoder_actor`, `encoder_critic`), gleiche Architektur
- Comm-Module — Actor und Critic haben jeweils **eigene** Comm-Layer

---

## 9. Visualisierung — Stufen-Pipeline

```
ENV
 │
 ▼
┌──────────────────────────────────────────────────────────────┐
│  Wrapper sammelt obs ALLER Agents zentral                    │
│  → für Agent i: (base_i, [obs_j, obs_k], tree_i)             │
│  → über 3 Zeitschritte: seq_i = [t-2, t-1, t]                │
└──────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────┐
│  Φ₁  ──  Temporal Attention (nur Self über Zeit)             │
│          base_i_{t-2,t-1,t}  ──►  h_temp                     │
└──────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────┐
│  Φ₂  ──  Tree wird auf Self draufaddiert                     │
│          h_temp + tree_encoder(tree_i)  ──►  h_self          │
└──────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────┐
│  Φ₃  ──  Spatial Attention: Self mischt mit Nachbarn         │
│          Q = h_self                                          │
│          K,V = [h_self, encoder(obs_j), encoder(obs_k)]      │
│                                          ──►  c              │
└──────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────┐
│  Φ₄  ──  Communication: gerichtete Messages                  │
│          messages = msg_proj(opp) + intent_embed             │
│          weights  = attention(send_k, recv_q) · gate         │
│          comm_vec = Σ messages · weights                     │
│                                          ──►  z              │
└──────────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────────┐
│  Heads  ──  Actor: π(a|s)     Critic: V(s)                   │
│             action = sample(π)                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Code-Referenzen

| Stufe | Methode / Klasse | Datei |
|---|---|---|
| Env-Obs | `DecisionPointObservation.get` | `marl_attention_temporal_observation/decision_point_observation.py` (≈L1303) |
| Wrapper | `TemporalMultiAgentObservation.get_many` | `marl_attention_temporal_observation/temporal_multi_agent_observation.py` (≈L164) |
| Φ₁ Temporal | `forward_agent` STEP 1+2 | `marl_attention_temporal_mappo.py` (≈L1080) |
| Φ₂ Tree | `_encode_tree_signal` + LN | `marl_attention_temporal_mappo.py` (≈L1126) |
| Φ₃ Spatial | `forward_agent` STEP 3 (Cross-Attention) | `marl_attention_temporal_mappo.py` (≈L1135) |
| Φ₄ Comm | `_apply_communication` | `marl_attention_temporal_mappo.py` (≈L1030) |
| Heads + Sampling | `_masked_act` (im DecisionPointPolicy) | `marl_attention_temporal.py` |

---

## 11. Referenzen

- TarMAC: Das et al. 2019 — *Targeted Multi-Agent Communication* (arXiv:1810.11187)
- DIAL: Foerster et al. 2016 — *Learning to Communicate with Deep MARL* (arXiv:1605.06676)
- CommNet: Sukhbaatar et al. 2016 (arXiv:1605.07736)
- MAPPO: Yu et al. 2022 (arXiv:2103.01955)
- Action Masking: Huang & Ontañón 2022 (arXiv:2006.14171)
