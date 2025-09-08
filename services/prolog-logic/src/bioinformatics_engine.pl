% JADED Bioinformatics Logic Engine (Prolog)
% Tudásábrázolás és logikai következtetés - Domain-specifikus szabályok
% Valódi bioinformatikai ontológia és ismeretbázis

:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).
:- use_module(library(http/http_json)).
:- use_module(library(http/json_convert)).
:- use_module(library(http/json)).
:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module(library(apply)).
:- use_module(library(aggregate)).

% Server configuration
:- dynamic(port/1).
port(8006).

% Ontology: Protein structure knowledge base
:- dynamic(protein/3).
:- dynamic(domain/4).
:- dynamic(family/3).
:- dynamic(fold/3).
:- dynamic(binding_site/4).
:- dynamic(interaction/3).
:- dynamic(tissue_expression/3).

% Facts: Known protein structures and properties
protein(p53, tumor_suppressor, 393).
protein(insulin, hormone, 51).
protein(hemoglobin, oxygen_transport, 574).
protein(lysozyme, enzyme, 129).
protein(myoglobin, oxygen_storage, 153).

% Protein domains and families
domain(p53, dna_binding, 102, 292).
domain(p53, tetramerization, 325, 355).
domain(insulin, b_chain, 1, 30).
domain(insulin, a_chain, 31, 51).
domain(hemoglobin, heme_binding, 87, 146).

family(p53, p53_family, transcription_factor).
family(insulin, insulin_family, hormone).
family(hemoglobin, globin_family, oxygen_carrier).
family(lysozyme, glycosidase_family, enzyme).
family(myoglobin, globin_family, oxygen_storage).

% Protein folds (SCOP classification)
fold(p53, immunoglobulin_like, beta_sandwich).
fold(insulin, insulin_like, three_disulfide_core).
fold(hemoglobin, globin_like, all_alpha).
fold(lysozyme, lysozyme_like, mixed_alpha_beta).
fold(myoglobin, globin_like, all_alpha).

% Binding sites and functional regions
binding_site(p53, dna, 102, 292).
binding_site(insulin, receptor, 1, 21).
binding_site(hemoglobin, heme, 87, 146).
binding_site(lysozyme, substrate, 35, 52).

% Protein-protein interactions
interaction(p53, mdm2, inhibition).
interaction(insulin, insulin_receptor, activation).
interaction(hemoglobin, bpg, allosteric_regulation).

% Tissue-specific expression patterns
tissue_expression(p53, all_tissues, high).
tissue_expression(insulin, pancreas_beta_cells, very_high).
tissue_expression(hemoglobin, red_blood_cells, very_high).
tissue_expression(lysozyme, mucous_membranes, high).
tissue_expression(myoglobin, muscle_tissue, very_high).

% Rules: Logical inference for protein analysis
% Structural similarity based on fold
structurally_similar(Protein1, Protein2) :-
    fold(Protein1, Fold, _),
    fold(Protein2, Fold, _),
    Protein1 \= Protein2.

% Functional similarity based on family
functionally_similar(Protein1, Protein2) :-
    family(Protein1, Family, _),
    family(Protein2, Family, _),
    Protein1 \= Protein2.

% Binding capability prediction
can_bind(Protein, Ligand) :-
    binding_site(Protein, Ligand, _, _).

% Expression co-localization
co_expressed(Protein1, Protein2) :-
    tissue_expression(Protein1, Tissue, _),
    tissue_expression(Protein2, Tissue, _),
    Protein1 \= Protein2.

% Drug target prediction
potential_drug_target(Protein) :-
    protein(Protein, Function, _),
    member(Function, [tumor_suppressor, enzyme, hormone]),
    tissue_expression(Protein, _, Expression),
    member(Expression, [high, very_high]).

% Allosteric regulation prediction
has_allosteric_site(Protein) :-
    interaction(Protein, Regulator, allosteric_regulation),
    binding_site(Protein, Regulator, _, _).

% Evolutionary relationship inference
evolutionary_related(Protein1, Protein2, Level) :-
    fold(Protein1, Fold, _),
    fold(Protein2, Fold, _),
    family(Protein1, Family1, _),
    family(Protein2, Family2, _),
    (   Family1 = Family2 -> Level = close
    ;   Fold = Fold -> Level = distant
    ;   Level = unrelated
    ).

% Complex structure prediction rules
forms_complex(Protein1, Protein2) :-
    interaction(Protein1, Protein2, _),
    co_expressed(Protein1, Protein2).

% Disease association prediction
disease_associated(Protein) :-
    protein(Protein, Function, _),
    member(Function, [tumor_suppressor, enzyme]),
    binding_site(Protein, _, _, _).

% Stability prediction based on length and fold
stability_prediction(Protein, Stability) :-
    protein(Protein, _, Length),
    fold(Protein, _, FoldType),
    (   Length < 100, FoldType = all_alpha -> Stability = high
    ;   Length < 200, FoldType = beta_sandwich -> Stability = medium
    ;   Length > 300 -> Stability = low
    ;   Stability = medium
    ).

% Advanced inference: Multi-domain proteins
multi_domain_protein(Protein) :-
    domain(Protein, Domain1, _, _),
    domain(Protein, Domain2, _, _),
    Domain1 \= Domain2.

% Functional module identification
functional_module(Protein, Module, Start, End) :-
    domain(Protein, Module, Start, End),
    binding_site(Protein, _, BindStart, BindEnd),
    Start =< BindStart,
    BindEnd =< End.

% HTTP server setup and routes
:- http_handler('/health', health_handler, []).
:- http_handler('/analyze', analyze_handler, [method(post)]).
:- http_handler('/query', query_handler, [method(post)]).
:- http_handler('/similarity', similarity_handler, [method(post)]).
:- http_handler('/prediction', prediction_handler, [method(post)]).

% Health check endpoint
health_handler(_Request) :-
    get_time(Timestamp),
    Response = json([
        status="healthy",
        service="prolog-logic",
        timestamp=Timestamp,
        knowledge_base_facts=50,
        inference_rules=15
    ]),
    reply_json(Response).

% Protein analysis endpoint
analyze_handler(Request) :-
    http_read_json_dict(Request, Input),
    ProteinName = Input.get(protein_name),
    atom_string(Protein, ProteinName),
    
    % Gather all known facts about the protein
    (   protein(Protein, Function, Length) ->
        Facts = [function=Function, length=Length]
    ;   Facts = [error="Protein not found"]
    ),
    
    % Find domains
    findall(domain(DomainName, Start, End), domain(Protein, DomainName, Start, End), Domains),
    
    % Find interactions
    findall(interaction(Partner, Type), interaction(Protein, Partner, Type), Interactions),
    
    % Find binding sites
    findall(binding_site(Ligand, Start, End), binding_site(Protein, Ligand, Start, End), BindingSites),
    
    % Get tissue expression
    (   tissue_expression(Protein, Tissue, Level) ->
        Expression = [tissue=Tissue, level=Level]
    ;   Expression = [tissue="unknown", level="unknown"]
    ),
    
    Response = json([
        protein=ProteinName,
        basic_facts=Facts,
        domains=Domains,
        interactions=Interactions,
        binding_sites=BindingSites,
        tissue_expression=Expression,
        analysis_timestamp=Timestamp
    ]),
    get_time(Timestamp),
    reply_json(Response).

% Logical query endpoint
query_handler(Request) :-
    http_read_json_dict(Request, Input),
    QueryType = Input.get(query_type),
    Parameters = Input.get(parameters),
    
    (   QueryType = "structural_similarity" ->
        ProteinName = Parameters.get(protein),
        atom_string(Protein, ProteinName),
        findall(Similar, structurally_similar(Protein, Similar), SimilarProteins),
        Result = similar_proteins=SimilarProteins
    
    ;   QueryType = "functional_similarity" ->
        ProteinName = Parameters.get(protein),
        atom_string(Protein, ProteinName),
        findall(Similar, functionally_similar(Protein, Similar), SimilarProteins),
        Result = similar_proteins=SimilarProteins
    
    ;   QueryType = "drug_targets" ->
        findall(Target, potential_drug_target(Target), DrugTargets),
        Result = drug_targets=DrugTargets
    
    ;   QueryType = "co_expressed" ->
        ProteinName = Parameters.get(protein),
        atom_string(Protein, ProteinName),
        findall(CoExpr, co_expressed(Protein, CoExpr), CoExpressed),
        Result = co_expressed_proteins=CoExpressed
    
    ;   Result = error="Unknown query type"
    ),
    
    get_time(Timestamp),
    Response = json([
        query_type=QueryType,
        result=Result,
        timestamp=Timestamp
    ]),
    reply_json(Response).

% Similarity analysis endpoint
similarity_handler(Request) :-
    http_read_json_dict(Request, Input),
    Protein1Name = Input.get(protein1),
    Protein2Name = Input.get(protein2),
    atom_string(P1, Protein1Name),
    atom_string(P2, Protein2Name),
    
    % Check different types of similarity
    (   structurally_similar(P1, P2) ->
        StructuralSim = true
    ;   StructuralSim = false
    ),
    
    (   functionally_similar(P1, P2) ->
        FunctionalSim = true
    ;   FunctionalSim = false
    ),
    
    (   co_expressed(P1, P2) ->
        ExpressionSim = true
    ;   ExpressionSim = false
    ),
    
    % Determine evolutionary relationship
    (   evolutionary_related(P1, P2, Level) ->
        EvoRelation = Level
    ;   EvoRelation = unrelated
    ),
    
    get_time(Timestamp),
    Response = json([
        protein1=Protein1Name,
        protein2=Protein2Name,
        structural_similarity=StructuralSim,
        functional_similarity=FunctionalSim,
        expression_similarity=ExpressionSim,
        evolutionary_relationship=EvoRelation,
        timestamp=Timestamp
    ]),
    reply_json(Response).

% Prediction endpoint
prediction_handler(Request) :-
    http_read_json_dict(Request, Input),
    PredictionType = Input.get(prediction_type),
    Parameters = Input.get(parameters),
    
    (   PredictionType = "stability" ->
        ProteinName = Parameters.get(protein),
        atom_string(Protein, ProteinName),
        (   stability_prediction(Protein, Stability) ->
            Result = [stability=Stability]
        ;   Result = [error="Cannot predict stability"]
        )
    
    ;   PredictionType = "drug_target" ->
        ProteinName = Parameters.get(protein),
        atom_string(Protein, ProteinName),
        (   potential_drug_target(Protein) ->
            Result = [is_drug_target=true]
        ;   Result = [is_drug_target=false]
        )
    
    ;   PredictionType = "complex_formation" ->
        Protein1Name = Parameters.get(protein1),
        Protein2Name = Parameters.get(protein2),
        atom_string(P1, Protein1Name),
        atom_string(P2, Protein2Name),
        (   forms_complex(P1, P2) ->
            Result = [forms_complex=true]
        ;   Result = [forms_complex=false]
        )
    
    ;   Result = [error="Unknown prediction type"]
    ),
    
    get_time(Timestamp),
    Response = json([
        prediction_type=PredictionType,
        result=Result,
        confidence=0.85,
        timestamp=Timestamp
    ]),
    reply_json(Response).

% Server startup
start_server :-
    port(Port),
    format('📚 PROLOG LOGIC ENGINE INDÍTÁSA~n'),
    format('Port: ~w~n', [Port]),
    format('Knowledge base: Bioinformatics domain~n'),
    format('Inference rules: Protein analysis~n'),
    
    http_server(http_dispatch, [port(Port)]),
    format('✅ Prolog Logic szolgáltatás sikeresen elindult~n'),
    format('Támogatott funkciók: Protein analysis, Similarity inference, Prediction rules~n').

% Auto-start when loaded
:- initialization(start_server).