# KI in Neurowissenschaften

## Einleitung
Das Seminar dreht sich um die Anwendung von Deep Learning im Bereich EEG. Momentan ist nichts im Bereich MRI geplant, wir können das aber gemeinsam diskutieren. Wir möchten mit dem Seminar einen Einstieg in dieses wirklich spannende Thema bieten und euch motivieren, auch abseits von der "klassischen" Bildgebung – also allem, was wirklich Bilder erzeugt – den Nutzen von Artificial Intelligence zu erkunden.

Unser Ziel ist es, euch möglichst viel Freiheit bei den Projekten zu bieten und gleichzeitig zu einem interessanten und sinnvollem Ergebnis zu kommen. Daher sind für die Projekte Aufgabenschwerpunkte beschrieben, die von euch bearbeitet werden sollen. Diese ähneln sich größtenteils für die unterschiedlichen Projekte. Aufgrund der unterschiedlichen Themen ist der Fokus jedoch zumeist leicht verschoben.

Im Folgenden findet ihr eine Beschreibung der Projekte. Schaut euch die Paper an (Abstracts lesen) und überlegt, was eurer Neigung entspricht. Ein Projekt kann auch von mehr als einem Team bearbeitet werden – es wäre aber toll, wenn nicht alle das Gleiche wählen.

Noch eine extra Anmerkung: Zu allen Projekten gibt es GitHub Repos. Wir empfehlen euch aber, die Implementierung zunächst selbst anzugehen. Das bringt euch mehr als Copy-and-paste am Ende. Solltet ihr nicht weiterkommen, versuchen wir euch zu unterstützen. Haben wir auch keine Idee, könnt ihr natürlich gerne auf die Repos zurückgreifen und schauen, wie es da gemacht wurde.


## Projekt 1 - Klassifikation von EEG-Signalen

### Motivation
Für Neurologen ist die Arbeit mit EEG-Daten ein unerlässliches Werkzeug. Ob bei Epilepsie-Patienten oder im Bereich der Schlafüberwachung. Um die EEG-Daten richtig
und vor allem verlässlich interpretieren zu können, bedarf es in der Regel Erfahrung und ein geschultes Auge. Um diese Arbeit in naher Zukunft zu vereinfachen und auch
bei komplexen EEG-Datensätzen eine verlässliche Einschätzung zu erhalten, ist der Einsatz von Machine Learning Algorithmen von Interesse. Da wir in diesem Seminar den
Fokus auf Deep Learning legen, ist das Ziel dieses Projektes Neuronale Netze einzusetzen, um Klassifikationsaufgaben zu bewältigen.

### Aufgabenbereiche

#### Datenaufbereitung
Eine wichtige Frage, die sich im Vorfeld stellt, ist die des Preprocessings. EEG-Messungen sind teuer hinsichtlich des zeitlichen Aufwands, weshalb viele Datensätzen nur wenige Teilnehmende haben. Daher sollt ihr in diesem Teil untersuchen, welche Möglichkeiten des Preprocessing es gibt, um aus einem kleinen Datensatz möglichst viele Trainingsbeispiele zu erzeugen. Des Weiteren sollt ihr den Einfluss verschiedener Preprocessing-Schritte wie z. B. dem Einsatz von Bandpass-Filtern auf das Ergebnis austesten. Eine Frage wäre z. B., ob ein Modell nur anhand eines kleinen Frequenzbandes lernen kann, um welche Aufgabe/ Handlung es sich bei dem durchgeführten Experiment gehandelt hat.

#### Modelle
Die Modelle, die von euch implementiert werden sollen, sind zunächst die Modelle des Papers (Q1). Es handelt sich hierbei ausschließlich um Convolutional Neural Networks. An dieser Stelle ein Hinweis: Alle Modelle sind in Braindecode enthalten. Nichtsdestotrotz soll versucht werden, Modelle und Parameter selbst einzustellen. Sollte dies nicht funktionieren, kann immer noch auf die bereits vorhandenen zurückgegriffen werden. Eine Implementierung von Modellen ist immer eine gute Übung.

#### Visualisierung der Entscheidungen
Diese Aufgabe zielt darauf ab, die Entscheidungen der Modelle nachzuvollziehen. Dafür sollt ihr im Wesentlichen zwei Ansätze verfolgen
1. Backpropagation mit Hinblick auf die Eingangsdaten
2. Visualisierung der einzelnen Layer-Outputs

Einen Startpunkt/ Hilfe bietet hier (Q.2). Das ist natürlich schon ordentlich vom Arbeitsaufwand her, aber vielleicht schafft ihr das ja!

#### (OPTIONAL) Über Convolution hinaus...
Abgesehen von den in den Papern genutzten Convolutional Neural Networks, existieren weitere Netzwerkarchitekturen. Für diesen Aufgabenteil sollt ihr entweder eine rekurrente Netzwerkstruktur oder Modelle mit Attention erproben. Ihr implementiert die Modelle am besten händisch. Es gibt in PyTorch aber z. B. auch einen Transformer, der angepasst werden kann.
Nach erfolgtem Hyperparameter Tuning könnt ihr die Ergebnisse mit den in Aufgabengebiet 2. erzeugten Resultaten vergleichen.

### Quellen
1. [Schirrmeister, Robin et. al.: Deep learning with convolutional neural networks for EEG decoding and visualization, Human Brain Mapping, Willey, 2017](https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730)
2. [Hartmann, Kay Gregor et. al.: Hierarchical internal representation of spectral features in deep convolutional networks trained for EEG decoding, arxiv, 2017](https://arxiv.org/abs/1711.07792)

## Projekt 2 - Self-Supervised Learning for EEG

### Motivation
EEG-Daten, insbesondere gelabelte, sind in der Regel teuer in der Beschaffung - oft fehlen Zeit, die nötige Anzahl an Probanden und auch geschultes Personal, welches die Daten mit Labeln versieht. Aus diesem Grund ist das Interesse an unüberwachtem (unsupervised) oder selbst-überwachtem, (self-supervised, SSL) Lernen für EEG, wie für medizinische Datensätze generell, groß.

Für dieses Projekt beschränken wir uns auf SSL. Das Ziel ist es, ein erstes Gefühl für das Thema zu bekommen. Die Literatur ist eher sparse in diesem Bereich und die Forschung an dem Thema steckt noch in den Kinderschuhen. Nichtsdestotrotz, ist es einen Blick wert.

### Aufgabenbereiche

#### Wahl des Datensets und Preprocessing
In den unten angegebenen Quellen (das Paper unter 2 ist die ausführliche Version) werden Methodiken präsentiert, um SSL auf EEG-Daten anzuwenden. Ihr sollt (aus Paper 2) zunächst eines der beiden Datensets wählen (geht dabei nach Interesse). Ihr verfolgt die Schritte zum Preprocessing und visualisiert das Ergebnis anhand eines Beispiels.

#### SSL - Techniken
Im Paper werden verschiedene SSL - Techniken vorgestellt. Das relative positioning sollt ihr einmal selbst implementieren und ausprobieren. Ihr könnt euch dann auch gerne noch ans temporal shuffling machen, solltet ihr genug Zeit machen.

#### Modelle
In Paper 2 werden einige Modelle vorgestellt. Die für uns wichtigen Architekturen sind die neuen SSL Modelle. Die Baselines oder auch Vergleichsmodelle sind nicht von Interesse. Wählt ein SSL-Modell und implementiert dieses. Trainiert das Modell mit dem im Paper angegeben Loss (ebenfalls selbst implementieren) und optimiert die Hyperparameter. 

#### Visualisierung
Die Visualisierung der Ergebnisse ist von Interesse. Ihr benötigt die PyRiemann Library, um die Latent Spaces nachzustellen. Ist das zu kompliziert oder zeitaufwendig, überlegt euch eigene Visualisierungen (visualisiert z. B. einzelne Dimensionen des latent space/ feature vectors). Das ist also ein etwas kreativer Part.

#### (Optional) Preprocessing Playground
Wenn ihr genug Zeit habt, probiert das Preprocessing zu ändern und erneut zu trainieren. Was ändert sich im latent space? Sind gewisse Preprocessing Schritte unentbehrlich? Was passiert mit den Feature-Vektoren? Sollte sich die Visualisierung als zu kompliziert herausstellen, könnte man Visualisierung und diesen Punkt tauschen.

### Quellen
1. [Banville, Hubert et. al: Self-Supervised Representation Learning from Electroencephalography Signals, IEEE Workshop on Machine Learning for Signal Processing, 2019](https://ieeexplore.ieee.org/xpl/conhome/8911118/proceeding)
2. [Banville, Hubert et. al: Uncovering the structure of clinical EEG signals with self-supervised learning, Journal of Neural Engineering, 2021](https://iopscience.iop.org/article/10.1088/1741-2552/abca18)

## Projekt 3 - Source Space Reconstruction

### Motivation
Eine der mitunter wichtigsten Fragen ist die nach der Quelle eines EEG-Signals. Da EEG-Messungen nicht-invasiv sind, messen wir nur, was als elektrische-und/oder magnetische Feldsignale durch den Schädel zu den Elektroden dringt. Der Schädel hat jedoch einen starken Einfluss darauf. Daher ist es sehr schwierig, auf Grundlage der Elektrodensignale, zu bestimmen, welches Areal im Hirn für das Signal verantwortlich ist - das *Source localisation problem* ist ill-posed.

Um eine Lösung für das inverse Problem zu finden, werden Methoden wie *Minimum Norm Estimation* (MNE) oder *eLoreta* eingesetzt. Als Alternative zu solchen "herkömmlichen" Verfahren, wurde kürzlich der Einsatz von Artificial Neural Networks (ANNs) zur Lösung des Problems untersucht (1). Ziel dieses Projektes ist es daher, sich mit der Lösung des inversen Problems auseinanderzusetzen und den Einsatz Neuronaler Netze zu erproben.

**Remark**: Für dieses Projekt gibt es keine vorgefertigte Library, welche die bereits trainierbaren Modelle enthält. Daher ist dieses Projekt eher explorativer Natur.

### Aufgabenbereiche

#### Simulation der Daten und Preprocessing
Wie bereits erklärt, sind EEG-Daten rar. Insbesonders, fehlen bei realen EEG-Datensätzen die "Ground Truth"-Daten, also eine Rekonstruktion der Quellen im Gehirn. Glücklicherweise, könnt ihr mit MNE die Daten erzeugen. Beispielhaft exisitert hierzu ein Notebook mit Notizen. Die Parameter findet ihr im Paper. Anschließend implementiert/ nutzt ihr die Preprocessing Schritte. Im besten Fall gehören diese zu eurer Datapipeline dazu.

#### Modelle
Genau wie bei allen anderen Projekten, sollt ihr auch hier das Modell aus dem Paper implementieren. Da das diejenigen, die bereits Introduction to DL oder weiterführendes gehört haben, nicht vor Probleme stellen wird, könnt ihr gerne eine Vergleichsarchitektur interpretieren – wie wäre es z. B. mit einer Art U-Net ;) .

#### Evaluierung
Die Ergebnisse des Trainings sowie die Predictions des Netzwerkes auf einem Testset können einfach und effizient visualisiert werden. Wie sehen eure Ergebnisse im Vergleich zum Paper aus?

### Quellen
1. [Hecker, Lukas et. Al.: ConvDip: A Convolutional Neural Network for Better EEG Source Imaging, Frontiers in Neuroscience, 2021](https://www.frontiersin.org/articles/10.3389/fnins.2021.569918/full)
2. [Michel, Christoph M.; Brunet, Denis: EEG Source Imaging: A Practical Review of the Analysis Steps, Frontiers in Neurology, 2019](https://www.frontiersin.org/articles/10.3389/fneur.2019.00325/full)
3. Tutorials: [Raw data simulation](https://mne.tools/stable/auto_examples/simulation/simulate_raw_data.html#sphx-glr-auto-examples-simulation-simulate-raw-data-py), [Forward modelling](https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py) ...

## Projekt 4 - Learning Interpretable Features from EEG-Recordings

### Motivation
Neuronale Netze (NNs) werden im Allgemeinen als "Black Boxes" angesehen. Die Entscheidungen von NNs sind zumeist nur schwer interpretierbar bzw. überhaupt zu visualisieren. Diese Tatsache kollidiert mit dem Bedarf nach interpretierbaren und sicheren Entscheidungen in anwendungsorientierten Bereich wie der Medizin oder dem autonomen Fahren. Die Arbeit mit EEG-Daten fällt hier genauso rein. Daher ist es Ziel dieses Projektes, auf Grundlage von Quelle 1, sich näher mit interpretierbaren und visualisierbaren Entscheidungen auseinanderzusetzen und auf ein Datenset anzuwenden.

### Aufgabenbereiche

#### Trainingsdaten und Vorbereitung
Basierend auf dem Paper von Prof. Stober, sollen die Daten für das Training vorbereitet werden. Es ist euch freigestellt, ein anderes Datenset zu wählen, MNE und Braindecode bieten euch andere Möglichkeiten (z.B. eines der Motor Imaging Sets). Ihr wendet das beschriebene Preprocessing auf euren gewählten Datensatz.

#### Trainingsstrategie
Das unten angegebene Paper zeichnet sich insbesonders durch die angewandten Trainingsstrategien aus. Das ist zum einen das Cross-Trial Encoding und zum anderen das Similarity-Constraint Encoding. Ersteres solltet ihr auf jeden Fall implementieren für euer Training. Wenn die Zeit dann ausreicht, versucht auch noch das Similarity-Constrain Encoding mit zu implementieren.

#### Modelle
Die Modelle sind allesamt Convolutional Neural Networks, dementsprechend ist die Implementierung straightforward. Ihr könnt euch auch gerne am Hydra-Net probieren. Allerdings solltet ihr bei Convolution bleiben, die Visualisierung ist da leichter.

#### Ergebnisse
Trainiert eure Modelle mit den obigen Trainingsstrategien. Fertigt anschließend auch solche "topographic Maps", um zu sehen, ob ihr Feature encoden konntet. Auch wenn ihr vielleicht keine direkte Interpretation findet (aufgrund mangelnder neurowissenschaftlicher Kenntnisse), könnten die Ergebnisse an sich konsistent sein.

### Quellen
1. [Stober, Sebastian et. al: Deep Feature Lerning for EEG Recordings, arxiv, 2016](https://arxiv.org/abs/1511.04306)

## Präsentation / Dokumentation (Alle)
Wenn ihr eine Note wollt, solltet ihr die Ergebnisse vorstellen. Dies geschieht entweder durch eine Abschlusspräsentation im Rahmen des Seminars oder durch Anfertigung eines Notebooks, an dem Sachen probiert werden können und ihr eine kleine Demo durchführt.

Im Falle der Präsentation sollten die folgenden drei Themengebiete vorgestellt werden. Solltet ihr euch für das interaktive Notebook entscheiden, könnt ihr den 1. Punkt im Notebook zum Nachlesen zusammenfassen.

#### Paper/ Theorie/ Relevanz (~10-15 Minuten)
Fasst die gelesenen Paper kurz zusammen und geht auf die wesentlichen Erkenntnisse ein. Welche Relevanz hat das Paper, was ist wichtiges Hintergrundwissen (Bereich Neuroscience, AI)?

#### Implementierung/ Umsetzung (~5 Minuten)
Hier stellt ihr eure Implementierung der Modelle, des Preprocessing und/ oder Visualisierung vor. Alles, was ihr benutzt habt, aber nicht selbst implementiert, solltet ihr zumindest benennen.

#### Ergebnisse (~10-15 Minuten)
Abschließen stellt ihr die erzielten Ergebnisse vor. Dies kann vergleichend (haben wir die Resultate der Paper erreicht/ verbessert etc.)oder auch rein informativ erfolgen (was sehen wir, was lernen wir daraus?). Besonderes Augenmerk sollte auf eine ansprechende und korrekte Darstellung gelegt werden.

#### Diskussion (~5-10 Minuten)
Die Teammitglieder beantworten Fragen und diskutieren ihre Ergebnisse mit den anderen Teams.

## Fragen
Habt ihr Fragen?


# Vorbereitung

Zur Vorbereitung auf das Seminar gibt es eine Phase des Selbststudiums (~20 Stunden). Die Inhalte des Selbststudiums sind für die ersten beiden Seminartage wichtig. Bitte stellt daher sicher, dass ihr genug Zeit für die Erarbeitung der Inhalte einplant.
Zur gegenseitigen Unterstützung werden Lerntandems gebildet.

## Einführung in KI und Neurowissenschaften
Für alle, die nochmal eine Einführung in Machine Learning und Neuronale Netze 
benötigen, gibt es die Möglichkeit auf die Inhalte des KI Campus zuzugreifen. Das ist auch gut für diejenigen von euch, die schon die entsprechenden Vorlesungen gehört haben, aber nochmal die "high-level" View haben wollen. In Teil 1 des Dr. Med KI geht es nochmal um die ML-Basics, in Teil 2 mehr um die Anwendung im medizinischen Bereich. Ihr werdet hier keine fancy Mathematik finden, dafür aber kurz und bündig was es zu wissen gilt. 

### Inhalte auf dem [KI-Campus](https://ki-campus.org/)
- [OER 1: Dr. med. KI](https://ki-campus.org/courses/drmedki2020): Einführung in KI im medizinischen Umfeld. Zielgruppe dieses OERs ist eher Einsteiger in die Thematik. Entscheidet selbst welche Inhalte für euch relevant und wichtig sind.
- [OER 2: Dr. med. KI - Fortgeschritten](https://ki-campus.org/courses/drmedki_experts) - Modul 1 + 4, auch Verfügbar als Podcast [Folge 8](https://drmedki.podigee.io/8-drmedki-folge8) und [Folge 9](https://drmedki.podigee.io/9-drmedki-folge9)

Um die Inhalte zu sehen müsst ihr euch einen Account auf der KI-Campus Seite anlegen. Mit dem Account könnt ihr dann nach der Veranstaltung auch weitere Inhalte des KI-Campus besuchen. Es sind einige sehr spannende Themen dabei.

Bei Fragen zu KI-Campus oder technischen Problemen mit der Anmeldung meldet euch bei Johannes.

Folgende Fragen solltet ihr nach dem durcharbeiten der Inhalte auf alle Fälle beantworten können:

- Maschinelles Lernen vs. Deep Learning? Ordnet ein.
- Regression vs. Classification...
- Was ist der Unterschied zwischen Supervised und Unsupervised Learning
- Was sind Anwendungsgebiete in der Medizin? 
- Wie sind Neuronale Netze aufgebaut, wie lernen Neuronale Netze? 

### Paper
Als Einführung in das Seminarthema sollt ihr einmal das Suvey Paper von Roy et al. - [Deep learning-based electroencephalography analysis: a systematic review](https://iopscience.iop.org/article/10.1088/1741-2552/ab260c/pdf)
durcharbeiten. Das Paper ist recht lang und eine Übersicht über die bereits vorhandene Literatur. Das ist gut um einen Überblick über die wesentlichen Anwendungsgebiete zu kommen. Gleichzeitig ist das Paper von 2019 und dementsprechend, in einem so jungen Forschungsbereich, doch schon recht alt. Unabhängig davon, solltet ihr das Paper aufmerksam lesen.

Um euch den Einstieg in den Bereich EEG zu erleichtern, sind hier noch 2 Quellen, die ein paar Grundlagen zusammenfassen. Unsere [1. Quelle](https://www.caam.rice.edu/~cox/wrap/eegwiki.pdf) ist der, ins pdf Format formatierte, Wikipedia Artikel zum Thema EEG. Ein Professor der Rice University in den USA hat den seinen Studenten gegeben - so schlecht kann der Artikel also nicht sein ;). Ihr könnt aber genauso gut den original Artikel auf Wikipedia (aber in Englisch!) nutzen, da findet ihr auch mehr Abbildungen. Die [2. Quelle](https://escholarship.org/content/qt5fb0q5wx/qt5fb0q5wx.pdf?t=owzywb) stammt von den Autoren, die eines der Standardwerke im Bereich EEG geschrieben haben. Wir hoffen, dass euch das hilft, die physikalischen Grundlagen sowie Anwendungen des EEG zu verstehen.

Folgende Fragen können euch helfen euer Verständnis für das Paper zu reflektieren:
- Was sind die medizinischen Anwendungsgebiete von EEG?
- In welchen dieser Anwendungsgebiete wird bereits intensiv mit AI gearbeitet?
- Was für Methoden (Netzwerkarchitekturen, supervised, unsupervised etc.) kommen im EEG-Anwendungsgebiet zum Einsatz?
- Was sind Schwächen der momentanen Ansätze? (Das ist auch eine Frage, für das Paper zur Source Reconstruction)
- Was ist das Forward Problem, was ist das inverse Problem?
- Wie unterscheiden sich die EEG - Anteile im Frequenzbereich, woran erkennt man sie? 
- Welche Themen sind noch offen/ unbeantwortet, wo kann mehr gemacht werden? 

Es gibt natürlich noch mehr Fragen, allerdings sollte das für den Einstieg reichen. Viel Spaß beim lesen! 


### Projekte
Wir haben schon eine Übersicht über die möglichen Kurzprojekte des Seminars erstellt.

In der initialen Selbststudiumsphase geht es noch nicht darum die Projekte zu lösen, vielmehr wollen wir, dass ihr euch einen ersten Überblick über die Projekte verschafft. 
Hierfür empfehlen wir euch die Projektbeschreibungen, sowie die Abstrakts der referenzierenten Paper durchzugehen. So könnt ihr für euch schon einmal eine Präferenz festlegen. Ihr findet die Projekte im Reiter.

### Entwicklungsumgebung
Im Seminar und in der Projektphase werden wir mit Google's Colab arbeiten.
Für die Benutzung von Colab braucht ihr einen Google Account. Wir haben im Repo ein Notebook für euch freigegeben. Dort findet ihr nochmal ein paar Grundlagen zu Python als auch zu PyTorch. Solltet ihr keine Erfahrungen mit Python oder Pytorch haben, empfehlen wir sehr dieses Notebook durchzuarbeiten, die kleinen Übungsaufgaben zu lösen und selbst etwas auszuprobieren. Kleinere Fragen können auch wir gerne beantworten, allerdings dient das Seminar nicht dazu, "Deep Learning zu lernen" oder eine Einführung in Python zu bekommen - da seid ihr selbst gefragt (und es gibt eine Menge an gutem Material im Internet). Den Link zum Repo findet ihr [hier](https://github.com/robert-DL/ai_neuroscience_ovgu/tree/main/tutorials). 

Neben Python und PyTorch kommen noch die Bibliotheken MNE und Braindecode zum Einsatz. Damit euch die Flut an APIs und Tutorials auf den Websiten nicht überrollt, werden wir euch im Seminar eine kleine Einführung geben. Für MNE wird es im wesentlichen um das Data Handling gehen. Für Braindecode gehen wir gemeinsam durch eines der Tutorials und schauen uns an, was man mit den Datasets und Anderen Sachen so machen kann. Das wird voraussichtlich alles am 1. Seminartag stattfinden. Wir werden die Notebooks dann auch später noch zum Repository hinzufügen, so dass ihr immer wieder reinschauen könnt. Hier findet ihr schon mal die Links zu den beiden Bibliotheken: 
- [MNE: EEG/ MEG Analyse und Visualisierung](https://mne.tools/stable/index.html)
- [Braindecode: EEG Decoding mit Deep Learning](https://braindecode.org/)
