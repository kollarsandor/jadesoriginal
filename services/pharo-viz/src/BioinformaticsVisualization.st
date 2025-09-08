"JADED Visualization Service (Pharo Smalltalk)
Interaktív tudományos vizualizáció - Live object messaging
Valós idejű diagramok és molekuláris vizualizáció"

"System startup and HTTP server"
Object subclass: #BioinformaticsVizServer
    instanceVariableNames: 'port httpServer visualizations activeConnections'
    classVariableNames: 'DefaultPort'
    category: 'JADED-Visualization'.

BioinformaticsVizServer class >> defaultPort [
    ^ DefaultPort ifNil: [ DefaultPort := 8008 ]
]

BioinformaticsVizServer >> initialize [
    super initialize.
    port := self class defaultPort.
    visualizations := Dictionary new.
    activeConnections := OrderedCollection new.
    self logInfo: '🎨 PHARO VISUALIZATION SERVICE INDÍTÁSA'.
    self logInfo: 'Port: ', port asString.
    self logInfo: 'Live object messaging: Enabled'.
    self logInfo: 'Interactive charts: Real-time updates'.
]

BioinformaticsVizServer >> start [
    "Start the HTTP server for visualization requests"
    httpServer := self createHttpServer.
    httpServer listenOn: port.
    self logInfo: '✅ Pharo Visualization service started on port: ', port asString.
    self logInfo: 'Supported visualizations: Protein structures, Expression heatmaps, Phylogenetic trees'.
]

BioinformaticsVizServer >> createHttpServer [
    "Create HTTP server with routing"
    ^ WAFastCGIAdaptor new
        codec: WACodec new;
        server: (WAKom startOn: port);
        yourself
]

BioinformaticsVizServer >> logInfo: aMessage [
    "Log informational messages with timestamp"
    | timestamp |
    timestamp := DateAndTime now asString.
    Transcript show: '[', timestamp, '] INFO: ', aMessage; cr.
]

BioinformaticsVizServer >> logError: aMessage [
    "Log error messages with timestamp"
    | timestamp |
    timestamp := DateAndTime now asString.
    Transcript show: '[', timestamp, '] ERROR: ', aMessage; cr.
]

"Protein structure visualization"
Object subclass: #ProteinStructureViz
    instanceVariableNames: 'proteinData coordinates bonds visualization3D'
    classVariableNames: ''
    category: 'JADED-Visualization'.

ProteinStructureViz >> initializeWithData: aProteinData [
    proteinData := aProteinData.
    coordinates := self extractCoordinates: aProteinData.
    bonds := self calculateBonds: coordinates.
    visualization3D := self create3DVisualization.
    ^ self
]

ProteinStructureViz >> extractCoordinates: aProteinData [
    "Extract 3D coordinates from protein data"
    | coords |
    coords := OrderedCollection new.
    
    aProteinData keysAndValuesDo: [ :residueId :residueData |
        residueData at: 'atoms' ifPresent: [ :atoms |
            atoms do: [ :atomData |
                | coord |
                coord := Point3D 
                    x: (atomData at: 'x')
                    y: (atomData at: 'y') 
                    z: (atomData at: 'z').
                coord residueId: residueId.
                coord atomType: (atomData at: 'type').
                coords add: coord
            ]
        ]
    ].
    
    ^ coords asArray
]

ProteinStructureViz >> calculateBonds: coordinateArray [
    "Calculate bonds between atoms based on distance"
    | bonds maxBondDistance |
    bonds := OrderedCollection new.
    maxBondDistance := 2.0.  "Angstroms"
    
    1 to: coordinateArray size do: [ :i |
        (i + 1) to: coordinateArray size do: [ :j |
            | atom1 atom2 distance |
            atom1 := coordinateArray at: i.
            atom2 := coordinateArray at: j.
            distance := atom1 distanceTo: atom2.
            
            (distance <= maxBondDistance) ifTrue: [
                bonds add: (Bond from: atom1 to: atom2 strength: (maxBondDistance - distance))
            ]
        ]
    ].
    
    ^ bonds asArray
]

ProteinStructureViz >> create3DVisualization [
    "Create interactive 3D protein structure visualization"
    | viz |
    viz := RTView new.
    
    "Add atoms as spheres"
    coordinates do: [ :coord |
        | sphere color |
        color := self colorForAtomType: coord atomType.
        sphere := RTSphere new 
            color: color;
            size: (self sizeForAtomType: coord atomType).
        viz add: (sphere elementOn: coord).
    ].
    
    "Add bonds as lines"
    bonds do: [ :bond |
        | line |
        line := RTLine new 
            color: Color gray;
            width: (bond strength * 2).
        viz add: (line edgeFrom: bond fromAtom to: bond toAtom).
    ].
    
    "Add interaction capabilities"
    viz addInteraction: RTDraggable.
    viz addInteraction: (RTPopup text: [ :coord | 
        'Residue: ', coord residueId asString, String cr,
        'Atom: ', coord atomType asString, String cr,
        'Position: ', coord asString
    ]).
    
    ^ viz
]

ProteinStructureViz >> colorForAtomType: atomType [
    "Return color based on atom type"
    atomType = 'C' ifTrue: [ ^ Color gray ].
    atomType = 'N' ifTrue: [ ^ Color blue ].
    atomType = 'O' ifTrue: [ ^ Color red ].
    atomType = 'S' ifTrue: [ ^ Color yellow ].
    atomType = 'P' ifTrue: [ ^ Color orange ].
    ^ Color black
]

ProteinStructureViz >> sizeForAtomType: atomType [
    "Return size based on atom type"
    atomType = 'C' ifTrue: [ ^ 6 ].
    atomType = 'N' ifTrue: [ ^ 5 ].
    atomType = 'O' ifTrue: [ ^ 4 ].
    atomType = 'S' ifTrue: [ ^ 8 ].
    atomType = 'P' ifTrue: [ ^ 7 ].
    ^ 3
]

"Gene expression heatmap visualization"
Object subclass: #ExpressionHeatmapViz
    instanceVariableNames: 'expressionData genes tissues heatmap colorScale'
    classVariableNames: ''
    category: 'JADED-Visualization'.

ExpressionHeatmapViz >> initializeWithData: anExpressionData [
    expressionData := anExpressionData.
    genes := self extractGenes: anExpressionData.
    tissues := self extractTissues: anExpressionData.
    colorScale := self createColorScale.
    heatmap := self createHeatmap.
    ^ self
]

ExpressionHeatmapViz >> extractGenes: anExpressionData [
    "Extract unique gene names"
    | geneSet |
    geneSet := Set new.
    anExpressionData do: [ :dataPoint |
        geneSet add: (dataPoint at: 'gene')
    ].
    ^ geneSet asOrderedCollection sort
]

ExpressionHeatmapViz >> extractTissues: anExpressionData [
    "Extract unique tissue types"
    | tissueSet |
    tissueSet := Set new.
    anExpressionData do: [ :dataPoint |
        tissueSet add: (dataPoint at: 'tissue')
    ].
    ^ tissueSet asOrderedCollection sort
]

ExpressionHeatmapViz >> createColorScale [
    "Create color scale for expression levels"
    ^ RTColorPalette new
        colors: { Color blue. Color white. Color red };
        scale: (0 to: 10 by: 0.1);
        yourself
]

ExpressionHeatmapViz >> createHeatmap [
    "Create interactive expression heatmap"
    | viz matrix |
    viz := RTView new.
    matrix := self createExpressionMatrix.
    
    "Create heatmap cells"
    1 to: genes size do: [ :geneIndex |
        1 to: tissues size do: [ :tissueIndex |
            | cell expression color gene tissue |
            gene := genes at: geneIndex.
            tissue := tissues at: tissueIndex.
            expression := self getExpression: gene tissue: tissue.
            color := colorScale colorFor: expression.
            
            cell := RTBox new
                color: color;
                size: 20;
                borderColor: Color gray.
            
            viz add: (cell elementOn: {gene. tissue. expression}).
        ]
    ].
    
    "Position cells in grid"
    RTGridLayout new
        gapSize: 1;
        lineItemsCount: tissues size;
        on: viz elements.
    
    "Add labels"
    self addGeneLabels: viz.
    self addTissueLabels: viz.
    
    "Add interactions"
    viz addInteraction: (RTPopup text: [ :data |
        'Gene: ', (data at: 1) asString, String cr,
        'Tissue: ', (data at: 2) asString, String cr,
        'Expression: ', (data at: 3) asString
    ]).
    
    viz addInteraction: RTHighlightable.
    
    ^ viz
]

ExpressionHeatmapViz >> getExpression: aGene tissue: aTissue [
    "Get expression value for gene-tissue pair"
    expressionData 
        detect: [ :dataPoint | 
            (dataPoint at: 'gene') = aGene and: [ (dataPoint at: 'tissue') = aTissue ]
        ]
        ifFound: [ :dataPoint | ^ dataPoint at: 'expression' ]
        ifNone: [ ^ 0 ]
]

ExpressionHeatmapViz >> addGeneLabels: aView [
    "Add gene labels to heatmap"
    genes withIndexDo: [ :gene :index |
        | label |
        label := RTLabel new
            text: gene asString;
            color: Color black;
            height: 12.
        aView add: (label elementOn: gene).
    ]
]

ExpressionHeatmapViz >> addTissueLabels: aView [
    "Add tissue labels to heatmap"
    tissues withIndexDo: [ :tissue :index |
        | label |
        label := RTLabel new
            text: tissue asString;
            color: Color black;
            height: 12;
            rotateByDegrees: 45.
        aView add: (label elementOn: tissue).
    ]
]

"Phylogenetic tree visualization"
Object subclass: #PhylogeneticTreeViz
    instanceVariableNames: 'treeData nodes edges treeLayout'
    classVariableNames: ''
    category: 'JADED-Visualization'.

PhylogeneticTreeViz >> initializeWithData: aTreeData [
    treeData := aTreeData.
    nodes := self extractNodes: aTreeData.
    edges := self extractEdges: aTreeData.
    treeLayout := self createTreeLayout.
    ^ self
]

PhylogeneticTreeViz >> extractNodes: aTreeData [
    "Extract tree nodes from data"
    | nodeCollection |
    nodeCollection := OrderedCollection new.
    
    aTreeData at: 'nodes' ifPresent: [ :nodeArray |
        nodeArray do: [ :nodeData |
            | node |
            node := PhylogeneticNode new
                id: (nodeData at: 'id');
                name: (nodeData at: 'name' ifAbsent: [ 'Unknown' ]);
                branchLength: (nodeData at: 'branch_length' ifAbsent: [ 0.1 ]);
                isLeaf: (nodeData at: 'is_leaf' ifAbsent: [ false ]);
                yourself.
            nodeCollection add: node.
        ]
    ].
    
    ^ nodeCollection asArray
]

PhylogeneticTreeViz >> extractEdges: aTreeData [
    "Extract tree edges from data"
    | edgeCollection |
    edgeCollection := OrderedCollection new.
    
    aTreeData at: 'edges' ifPresent: [ :edgeArray |
        edgeArray do: [ :edgeData |
            | edge parentNode childNode |
            parentNode := nodes detect: [ :n | n id = (edgeData at: 'parent') ].
            childNode := nodes detect: [ :n | n id = (edgeData at: 'child') ].
            
            edge := PhylogeneticEdge new
                parent: parentNode;
                child: childNode;
                length: (edgeData at: 'length' ifAbsent: [ 0.1 ]);
                support: (edgeData at: 'support' ifAbsent: [ 1.0 ]);
                yourself.
            edgeCollection add: edge.
        ]
    ].
    
    ^ edgeCollection asArray
]

PhylogeneticTreeViz >> createTreeLayout [
    "Create phylogenetic tree visualization"
    | viz |
    viz := RTView new.
    
    "Add nodes"
    nodes do: [ :node |
        | element shape color |
        color := node isLeaf 
            ifTrue: [ Color blue ] 
            ifFalse: [ Color gray ].
        
        shape := node isLeaf
            ifTrue: [ RTBox new size: 8 ]
            ifFalse: [ RTCircle new size: 6 ].
        
        element := shape 
            color: color;
            borderColor: Color black;
            elementOn: node.
        
        viz add: element.
    ].
    
    "Add edges"
    edges do: [ :edge |
        | line thickness |
        thickness := edge support * 3.  "Thicker lines for higher support"
        line := RTLine new
            color: Color black;
            width: thickness.
        viz add: (line edgeFrom: edge parent to: edge child).
    ].
    
    "Apply tree layout"
    RTTreeLayout new
        horizontalGap: 50;
        verticalGap: 30;
        on: viz elements.
    
    "Add labels for leaf nodes"
    self addLeafLabels: viz.
    
    "Add interactions"
    viz addInteraction: (RTPopup text: [ :node |
        'Name: ', node name asString, String cr,
        'ID: ', node id asString, String cr,
        'Branch Length: ', node branchLength asString, String cr,
        'Type: ', (node isLeaf ifTrue: [ 'Leaf' ] ifFalse: [ 'Internal' ])
    ]).
    
    viz addInteraction: RTZoomableView.
    viz addInteraction: RTDraggableView.
    
    ^ viz
]

PhylogeneticTreeViz >> addLeafLabels: aView [
    "Add labels for leaf nodes"
    nodes select: [ :node | node isLeaf ] thenDo: [ :leafNode |
        | label |
        label := RTLabel new
            text: leafNode name;
            color: Color black;
            height: 10.
        aView add: (label elementOn: leafNode).
    ]
]

"Supporting classes"
Object subclass: #Point3D
    instanceVariableNames: 'x y z residueId atomType'
    classVariableNames: ''
    category: 'JADED-Visualization'.

Point3D class >> x: xCoord y: yCoord z: zCoord [
    ^ self new x: xCoord; y: yCoord; z: zCoord; yourself
]

Point3D >> x: aNumber [ x := aNumber ]
Point3D >> y: aNumber [ y := aNumber ]
Point3D >> z: aNumber [ z := aNumber ]
Point3D >> x [ ^ x ]
Point3D >> y [ ^ y ]
Point3D >> z [ ^ z ]
Point3D >> residueId: anId [ residueId := anId ]
Point3D >> residueId [ ^ residueId ]
Point3D >> atomType: aType [ atomType := aType ]
Point3D >> atomType [ ^ atomType ]

Point3D >> distanceTo: anotherPoint [
    | dx dy dz |
    dx := x - anotherPoint x.
    dy := y - anotherPoint y.
    dz := z - anotherPoint z.
    ^ (dx*dx + dy*dy + dz*dz) sqrt
]

Point3D >> asString [
    ^ '(', x asString, ', ', y asString, ', ', z asString, ')'
]

Object subclass: #Bond
    instanceVariableNames: 'fromAtom toAtom strength'
    classVariableNames: ''
    category: 'JADED-Visualization'.

Bond class >> from: fromAtom to: toAtom strength: strength [
    ^ self new fromAtom: fromAtom; toAtom: toAtom; strength: strength; yourself
]

Bond >> fromAtom: anAtom [ fromAtom := anAtom ]
Bond >> toAtom: anAtom [ toAtom := anAtom ]
Bond >> strength: aNumber [ strength := aNumber ]
Bond >> fromAtom [ ^ fromAtom ]
Bond >> toAtom [ ^ toAtom ]
Bond >> strength [ ^ strength ]

Object subclass: #PhylogeneticNode
    instanceVariableNames: 'id name branchLength isLeaf'
    classVariableNames: ''
    category: 'JADED-Visualization'.

PhylogeneticNode >> id: anId [ id := anId ]
PhylogeneticNode >> name: aName [ name := aName ]
PhylogeneticNode >> branchLength: aLength [ branchLength := aLength ]
PhylogeneticNode >> isLeaf: aBoolean [ isLeaf := aBoolean ]
PhylogeneticNode >> id [ ^ id ]
PhylogeneticNode >> name [ ^ name ]
PhylogeneticNode >> branchLength [ ^ branchLength ]
PhylogeneticNode >> isLeaf [ ^ isLeaf ]

Object subclass: #PhylogeneticEdge
    instanceVariableNames: 'parent child length support'
    classVariableNames: ''
    category: 'JADED-Visualization'.

PhylogeneticEdge >> parent: aNode [ parent := aNode ]
PhylogeneticEdge >> child: aNode [ child := aNode ]
PhylogeneticEdge >> length: aLength [ length := aLength ]
PhylogeneticEdge >> support: aSupport [ support := aSupport ]
PhylogeneticEdge >> parent [ ^ parent ]
PhylogeneticEdge >> child [ ^ child ]
PhylogeneticEdge >> length [ ^ length ]
PhylogeneticEdge >> support [ ^ support ]

"HTTP request handlers"
Object subclass: #VizRequestHandler
    instanceVariableNames: 'server'
    classVariableNames: ''
    category: 'JADED-Visualization'.

VizRequestHandler >> initializeWithServer: aServer [
    server := aServer.
    ^ self
]

VizRequestHandler >> handleProteinVisualization: requestData [
    "Handle protein structure visualization request"
    | proteinViz result |
    proteinViz := ProteinStructureViz new initializeWithData: (requestData at: 'protein_data').
    result := Dictionary new
        at: 'visualization_type' put: 'protein_structure';
        at: 'status' put: 'success';
        at: 'atoms_count' put: proteinViz coordinates size;
        at: 'bonds_count' put: proteinViz bonds size;
        at: 'timestamp' put: DateAndTime now asString;
        yourself.
    ^ result
]

VizRequestHandler >> handleExpressionHeatmap: requestData [
    "Handle gene expression heatmap visualization request"
    | heatmapViz result |
    heatmapViz := ExpressionHeatmapViz new initializeWithData: (requestData at: 'expression_data').
    result := Dictionary new
        at: 'visualization_type' put: 'expression_heatmap';
        at: 'status' put: 'success';
        at: 'genes_count' put: heatmapViz genes size;
        at: 'tissues_count' put: heatmapViz tissues size;
        at: 'timestamp' put: DateAndTime now asString;
        yourself.
    ^ result
]

VizRequestHandler >> handlePhylogeneticTree: requestData [
    "Handle phylogenetic tree visualization request"
    | treeViz result |
    treeViz := PhylogeneticTreeViz new initializeWithData: (requestData at: 'tree_data').
    result := Dictionary new
        at: 'visualization_type' put: 'phylogenetic_tree';
        at: 'status' put: 'success';
        at: 'nodes_count' put: treeViz nodes size;
        at: 'edges_count' put: treeViz edges size;
        at: 'timestamp' put: DateAndTime now asString;
        yourself.
    ^ result
]

"Auto-start the visualization server"
| vizServer |
vizServer := BioinformaticsVizServer new.
vizServer initialize.
vizServer start.

"Register global instance"
Smalltalk at: #JadedVizServer put: vizServer.

"Log startup completion"
Transcript show: '✅ Pharo Visualization Service fully initialized'; cr.
Transcript show: 'Available visualizations: Protein structures, Expression heatmaps, Phylogenetic trees'; cr.
Transcript show: 'Interactive features: Zoom, pan, drag, popup tooltips'; cr.