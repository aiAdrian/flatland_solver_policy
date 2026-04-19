#### Erweiterung für Merge/Kreuzung (decision_type==3)

Für Situationen, in denen sich ein Agent eine Zelle vor einer Weiche, Einmündung oder Kreuzung befindet (decision_type==3), wurde die Beobachtung um folgende Logik erweitert:

- **Vorwärts-Analyse:** Es wird geprüft, ob das Feld vor dem Agenten frei ist. Ist das Feld belegt, wird zusätzlich geprüft, ob der belegende Agent aus der Gegenrichtung kommt (Kreuzungs-/Deadlock-Risiko).
- **Deadlock-Erkennung:** Kommt ein Agent entgegen, wird ein Deadlock-Flag gesetzt. Dies signalisiert, dass ein Einfädeln oder Kreuzen aktuell riskant ist.
- **Warte-Flag:** Für das Feld hinter dem Agenten (Gegenrichtung) wird geprüft, ob dort ein Agent steht, der auf den Agenten zufährt. In diesem Fall wird ein Warte-Flag gesetzt, um dem anderen Agenten Vorrang zu geben und Konflikte zu vermeiden.
- **Feature-Speicherung:** Die ermittelten Flags (Feld vorwärts frei, entgegenkommender Agent, Deadlock, Warten) werden explizit als Features im Beobachtungsvektor abgelegt.

Diese Erweiterung ermöglicht Policies, an Einmündungen und Kreuzungen situationsabhängig zu entscheiden, ob ein Einfädeln, Kreuzen oder Warten sinnvoll ist, und so Deadlocks und Konflikte proaktiv zu vermeiden.
### Erweiterte DecisionPointObservation (2026)

Die DecisionPointObservation wurde um eine fortschrittliche Navigations- und Deadlock-Logik erweitert, die speziell für Agenten an Weichen entwickelt wurde:

- **Richtungsbasierte Analyse:** Für jede mögliche Richtung an einer Weiche (links, geradeaus, rechts, rückwärts) wird individuell geprüft, wie der Agent zum Ziel gelangen kann.
- **Navigation und Deadlock-Erkennung:** Der Agent läuft entlang des Pfads in die gewählte Richtung, bis er entweder das Ziel erreicht, auf einen entgegenkommenden Agenten trifft oder kein Weg mehr möglich ist.
- **Backtracking:** Trifft der Agent auf einen entgegenkommenden Agenten, wird zur letzten Weiche zurückgegangen (Backtracking) und dort versucht, eine alternative Richtung zu wählen. Dies geschieht rekursiv, bis ein Weg zum Ziel gefunden wird oder alle Alternativen erschöpft sind.
- **Feature-Berechnung pro Richtung:** Für jede Richtung werden drei Werte berechnet und als Features gespeichert:
	- Die Distanz zum Ziel (bzw. -1, falls kein Weg gefunden wurde)
	- Ein Deadlock-Flag (1, falls Deadlock, sonst 0)
	- Die Anzahl der Pfadwechsel (wie oft beim Backtracking eine neue Richtung gewählt wurde)
- **Kürzester Pfad:** Die Navigation folgt immer dem aktuell kürzesten Pfad zum Ziel, sofern kein Deadlock oder Blockade vorliegt.
- **Robuste Entscheidungsgrundlage:** Diese Logik ermöglicht es Policies, nicht nur die aktuelle Situation, sondern auch potenzielle Konflikte und Ausweichmöglichkeiten an Weichen explizit zu berücksichtigen.

Durch diese Erweiterung ist die DecisionPointObservation besonders mächtig für Multi-Agenten-Szenarien mit komplexen Weichen- und Konfliktsituationen.
# Übersicht der vier Observations in `temporal_multi_agent_observation`

In diesem Modul werden vier verschiedene Beobachtungsarten (Observations) für Multi-Agenten-Umgebungen im Flatland-Rail-Setting implementiert. Im Folgenden werden diese vier Observations in Prosa beschrieben:

## Basisklassen und Alternativen

Neben den temporalen Observations gibt es drei wichtige Basisklassen, die als Grundlage oder Alternative für die zeitlichen Varianten dienen:


### 1. `ExperimentalObservation`
Diese Beobachtung liefert eine umfassende, 30-dimensionale Feature-Vector-Repräsentation für jeden Agenten. Sie kombiniert eigene Agenten-Informationen (Position, Richtung, Ziel, Status) mit einer Analyse der Umgebung und anderer Agenten. Die wichtigsten Merkmale:

- **Feature-Design:** 30D-Vektor pro Agent, der u.a. Entscheidungsstellen, Weichen, Hindernisse und andere Agenten abbildet.
- **Entscheidungsstellen:** Nutzt die `DecisionPointObservation` zur Analyse von Weichen, Merges und Kreuzungen.
- **Multi-Agent-Features:** Berücksichtigt andere Agenten in der Umgebung, z.B. für Konflikt- und Deadlock-Erkennung.
- **Pfad- und Zielanalyse:** Integriert Distanz- und Richtungsinformationen zum Ziel.
- **Reset-Logik:** Initialisiert Hilfsobjekte wie den `RailroadSwitchAnalyser` und einen Walker für Pfadanalysen.
- **get_many:** Baut eine Agenten-Karte und liefert für alle Agenten die eigenen Features (ohne Multi-Agent-Padding).

Diese Observation ist besonders geeignet als Basis für komplexere, temporale oder multi-agentenfähige Beobachtungen.



### 2. `DecisionPointObservation`

Diese Observation ist speziell auf die drei wichtigsten Entscheidungssituationen im Flatland-Setting zugeschnitten:

1. **Start (READY_TO_DEPART):** Der Agent entscheidet, ob er auf das Spielfeld fährt.
2. **An einer Weiche:** Der Agent steht auf einer Weiche und kann abzweigen (klassische Routing-Entscheidung, mehrere Pfade möglich).
3. **Vor einer Weiche (Merge/Crossing):** Der Agent steht ein Feld vor einer Weiche, kann nicht abzweigen, aber es besteht Konfliktpotenzial durch Einfädeln oder Kreuzen.

**Feature-Design und Logik:**
- Die Observation liefert einen kompakten, aber sehr informationsreichen Feature-Vektor, der alle für die Entscheidung relevanten Aspekte abbildet.
- Die ersten Features kodieren den Entscheidungstyp (Start, Switch, Merge), den Agentenzustand, Richtung, Position und Ziel.
- Es werden explizite Flags für Weichen, Merge-Situationen, Deadlocks und Wartebedarf gesetzt.
- Die Analyse nutzt den `RailroadSwitchAnalyser` zur präzisen Erkennung von Weichen, Einmündungen und Kreuzungen.

**Speziallogik für Weichen (decision_type==2):**
- Für jede mögliche Richtung (links, geradeaus, rechts, rückwärts) wird individuell geprüft, wie der Agent zum Ziel gelangen kann.
- Die Navigation folgt dem kürzesten Pfad, prüft aber auch auf entgegenkommende Agenten und Deadlock-Risiken.
- Trifft der Agent auf einen entgegenkommenden Agenten, wird zur letzten Weiche zurückgegangen (Backtracking) und dort versucht, eine alternative Richtung zu wählen. Die Anzahl der Pfadwechsel wird gezählt.
- Für jede Richtung werden drei Werte als Features gespeichert: Distanz zum Ziel (bzw. -1, falls kein Weg gefunden wurde), Deadlock-Flag (1, falls Deadlock, sonst 0), Anzahl der Pfadwechsel.


**Speziallogik für Merge/Kreuzung (decision_type==3):**
- Die Analyse für das Feld vor (forward) und hinter (backward) dem Agenten erfolgt jeweils mit der rekursiven Methode `_navigate_direction`, die Deadlocks, entgegenkommende Agenten und Pfadblockaden erkennt.
- Für beide Richtungen wird geprüft, ob die Zielzelle überhaupt im Grid liegt (Grid-Boundary-Check). Ist dies nicht der Fall, wird sofort ein Deadlock-Flag gesetzt und die Features entsprechend belegt.
- Für das Feld vor dem Agenten (forward) werden folgende Features gesetzt:
	- `forward_free`: 1, wenn ein Pfad zum Ziel existiert, sonst 0
	- `forward_agent`: 1, wenn auf dem Pfad ein fremder Agent gesehen wird, sonst 0
	- `forward_deadlock`: 1, wenn ein Deadlock erkannt wurde, sonst 0
- Für das Feld hinter dem Agenten (backward) wird mit `_navigate_direction` geprüft, ob ein Agent entgegenkommt. Falls ja, wird das `wait_flag` gesetzt (1), um Vorrang zu geben.
- Die Features werden explizit im Beobachtungsvektor abgelegt (forward_free, forward_agent, forward_deadlock, wait_flag).
- Durch die Grid-Boundary-Prüfung werden Indexierungsfehler und Out-of-Bounds-Zugriffe sicher verhindert.

Diese robuste und rekursive Analyse ermöglicht Policies, an Einmündungen und Kreuzungen situationsabhängig zu entscheiden, ob ein Einfädeln, Kreuzen oder Warten sinnvoll ist, und so Deadlocks und Konflikte proaktiv zu vermeiden.

**Zusammenfassung:**
- Die DecisionPointObservation ist ideal für Policies, die gezielt an Knotenpunkten, Weichen und Einmündungen agieren müssen.
- Sie bietet eine robuste Entscheidungsgrundlage für Multi-Agenten-Szenarien mit komplexen Konflikt- und Deadlock-Situationen.


### 3. `SimplifiedPathThreeTierObservation`
Diese Beobachtung teilt die Umgebung eines Agenten in drei Pfadsegmente ("Tiers") auf:

- **A: Links** (nur aktiv bei Weiche und wenn ein Pfad nach links existiert)
- **B: Geradeaus** (immer aktiv, enthält Features für den direkten Weg)
- **C: Rechts** (nur aktiv bei Weiche und wenn ein Pfad nach rechts existiert)

**Feature-Design:**
- Jeder Pfadteil hat einen eigenen Feature-Vektor (z.B. 16D), ergänzt um einen Header (z.B. 9D) mit State, Richtung, Ziel, Agentenzahl und Pfad-Hinweisen.
- Für jeden Pfad werden Distanz zum Ziel, Agenten auf dem Pfad, entgegenkommende Agenten, Deadlock- und Switch-Flags, Crowding, Pfadlänge, Blockaden und Zielerreichung berechnet.
- Deadlock-Detection: Erkennt entgegenkommende Agenten bis zur nächsten Weiche und prüft, ob ein Deadlock möglich ist.
- Crowd-Feature: Misst die Agentendichte im Umkreis entlang des Pfads.
- Optional: Path D (Rückfluss) für Überhol- und Rückzugsanalysen.

**Besonderheiten:**
- Die Beobachtung ist modular und kann für verschiedene Richtungen und Pfadtypen angepasst werden.
- Sie eignet sich besonders für Policies, die situationsabhängig zwischen mehreren Alternativen (z.B. Abzweigen, Geradeaus, Rückzug) wählen müssen.

## 1. `TemporalMultiAgentObservation`
Diese Observation kombiniert die Beobachtungen mehrerer Agenten über verschiedene Zeitschritte hinweg. Jeder Agent erhält nicht nur die aktuelle Umgebung, sondern auch eine Historie seiner eigenen Beobachtungen. Dadurch kann der Agent zeitliche Zusammenhänge und Veränderungen in der Umgebung besser erfassen und für seine Entscheidungsfindung nutzen. Die Beobachtung ist besonders nützlich für Algorithmen, die auf zeitlichen Abhängigkeiten basieren, wie z.B. rekurrente neuronale Netze.

## 2. `TemporalMultiAgentSwitchObservation`
Hierbei handelt es sich um eine spezialisierte Variante der temporalen Multi-Agenten-Observation, die den Fokus auf Weichen (Switches) im Schienennetz legt. Die Beobachtung enthält explizite Informationen über die Positionen und Zustände von Weichen, sowohl für den aktuellen als auch für vergangene Zeitschritte. Dies ermöglicht es den Agenten, strategische Entscheidungen an kritischen Knotenpunkten im Schienennetz zu treffen, insbesondere im Hinblick auf Konfliktvermeidung und effiziente Routenwahl.

## 3. `TemporalMultiAgentSwitchCellObservation`
Diese Observation erweitert die vorherige, indem sie nicht nur die Weichen selbst, sondern auch die Zellen, die zu Weichen führen oder in deren Nähe liegen, explizit berücksichtigt. Die Agenten erhalten Informationen darüber, ob sie sich auf einer Weichenzelle befinden oder sich einer solchen nähern. Dies ist besonders relevant für das Warten vor Weichen, das Überholen und die Lösung von Konflikten zwischen Agenten.

## 4. `TemporalMultiAgentSwitchCellWithDirectionObservation`
Diese Beobachtung ergänzt die `SwitchCellObservation` um Richtungsinformationen. Die Agenten erhalten nicht nur die Information, ob sie sich auf einer Weichenzelle oder in deren Nähe befinden, sondern auch, in welche Richtung sie sich bewegen und welche Abzweigungen in ihrer aktuellen Richtung möglich sind. Dadurch können Agenten noch gezielter planen, wann und wie sie an Weichen abbiegen oder warten sollten, um Kollisionen und Staus zu vermeiden.

---

Jede dieser Observations ist darauf ausgelegt, die Entscheidungsfindung der Agenten in komplexen, dynamischen Schienennetzen zu verbessern, indem sie relevante Umgebungsinformationen über die Zeit und im Kontext von Weichen bereitstellt.