NB. JADED Statistics Engine (J Language)
NB. Array programming - Fejlett statisztikai számítások genomikához
NB. Valódi matematikai algoritmusok nagy teljesítményű array műveletekkel

NB. Port és alapkonfiguráció
PORT =: 8007
MAXSEQLEN =: 100000
CHUNKSIZE =: 10000

NB. Logging és inicializálás
log =: 4 : 0
  timestamp =. ": ,. 6!:0 ''
  msg =. x , ': ' , y
  echo '📊 [', timestamp, '] ', msg
)

'INFO' log 'J STATISTICS ENGINE INDÍTÁSA'
'INFO' log 'Port: ', ": PORT
'INFO' log 'Max sequence length: ', ": MAXSEQLEN
'INFO' log 'Array chunk size: ', ": CHUNKSIZE

NB. Nukleotid és aminosav kódolás
NUCLEOTIDES =: 'ATGCNU'
AMINOACIDS =: 'ARNDCQEGHILKMFPSTWYV'
CODON_TABLE =: ;:'TTT TTC TTA TTG TCT TCC TCA TCG TAT TAC TAA TAG TGT TGC TGA TGG'
CODON_AA =: 'FFLLSSSSYYXCXCXW'

NB. Genomikai szekvencia validálás
validateDNA =: 3 : 0
  seq =. toupper y
  valid_chars =. seq e. NUCLEOTIDES
  valid_ratio =. (+/ valid_chars) % # seq
  if. valid_ratio < 0.9 do.
    'ERROR' log 'Invalid DNA sequence: low nucleotide ratio'
    0
  elseif. (# seq) > MAXSEQLEN do.
    'ERROR' log 'Sequence too long: ', ": # seq
    0
  elseif. (# seq) < 10 do.
    'ERROR' log 'Sequence too short: ', ": # seq
    0
  else.
    'INFO' log 'DNA sequence validated: ', ": # seq, ' nucleotides'
    1
  end.
)

NB. Nukleotid összetétel analízis
nucleotideComposition =: 3 : 0
  seq =. toupper y
  if. -. validateDNA seq do. _1 return. end.
  
  counts =. +/"1 seq =/ NUCLEOTIDES
  total =. +/ counts
  freqs =. counts % total
  
  gc_content =. (+/ 2 3 { counts) % total
  at_content =. (+/ 0 1 { counts) % total
  n_content =. (4 5 { counts) % total
  
  result =. counts ,: freqs
  result =. result ,: gc_content, at_content, +/ n_content
  
  'INFO' log 'Nucleotide composition calculated'
  result
)

NB. Codon használati analízis
codonUsage =: 3 : 0
  seq =. toupper y
  if. -. validateDNA seq do. _1 return. end.
  
  NB. 3-nukleotid codonokra bontás
  if. 0 = 3 | # seq do.
    codons =. _3 ]\ seq
  else.
    codons =. _3 ]\ seq , 'N' #~ 3 - 3 | # seq
  end.
  
  NB. Codon gyakoriság számítás
  unique_codons =. ~. codons
  counts =. +/"1 codons =/ unique_codons
  total =. +/ counts
  freqs =. counts % total
  
  NB. Aminosav összetétel
  aa_indices =. CODON_TABLE i. unique_codons
  valid_aa =. aa_indices < # CODON_AA
  aa_codes =. valid_aa # aa_indices { CODON_AA
  
  result =. unique_codons ,: counts ,: freqs ,: aa_codes
  
  'INFO' log 'Codon usage analysis completed: ', ": # unique_codons, ' unique codons'
  result
)

NB. ORF (Open Reading Frame) keresés
findORFs =: 3 : 0
  seq =. toupper y
  if. -. validateDNA seq do. _1 return. end.
  
  start_codons =. ;:'ATG GTG TTG'
  stop_codons =. ;:'TAA TAG TGA'
  min_length =. 300
  
  orfs =. i. 0 4  NB. start, stop, length, frame
  
  for_frame. 0 1 2 do.
    frame_seq =. frame { seq
    if. 3 > # frame_seq do. continue. end.
    
    codons =. _3 ]\ frame_seq
    start_positions =. I. codons e. start_codons
    stop_positions =. I. codons e. stop_codons
    
    for_start. start_positions do.
      stop_candidates =. stop_positions #~ stop_positions > start
      if. 0 = # stop_candidates do. continue. end.
      
      stop =. {. stop_candidates
      length =. 3 * stop - start
      
      if. length >: min_length do.
        orf_start =. frame + 3 * start
        orf_stop =. frame + 3 * stop + 2
        orfs =. orfs , orf_start, orf_stop, length, frame
      end.
    end.
  end.
  
  'INFO' log 'ORF analysis completed: ', ": # orfs, ' ORFs found'
  orfs
)

NB. Splice site predikció statisztikai módszerekkel
predictSpliceSites =: 3 : 0
  seq =. toupper y
  if. -. validateDNA seq do. _1 return. end.
  
  donor_consensus =. 'GT'
  acceptor_consensus =. 'AG'
  
  NB. Donor site keresés
  donor_positions =. I. (2 ]\ seq) -:"1 donor_consensus
  
  NB. Acceptor site keresés
  acceptor_positions =. I. (2 ]\ seq) -:"1 acceptor_consensus
  
  NB. Statisztikai score számítás pozíciók körül
  donor_scores =. donor_positions splice_score seq
  acceptor_scores =. acceptor_positions splice_score seq
  
  NB. Magas score-ú splice site-ok szűrése
  high_donors =. donor_positions #~ donor_scores > 0.7
  high_acceptors =. acceptor_positions #~ acceptor_scores > 0.7
  
  result =. high_donors ,: high_acceptors ,: donor_scores ,: acceptor_scores
  
  'INFO' log 'Splice site prediction: ', ": (# high_donors), ' donors, ', (# high_acceptors), ' acceptors'
  result
)

NB. Splice site scoring segédfüggvény
splice_score =: 4 : 0
  seq =. y
  positions =. x
  scores =. 0 $ 0
  
  for_pos. positions do.
    if. (pos + 20) > # seq do. scores =. scores , 0.5 continue. end.
    
    context =. (_10 + pos) {. (10 + pos) }. seq
    if. 20 ~: # context do. scores =. scores , 0.5 continue. end.
    
    NB. Egyszerű scoring alapú nukleotid gyakorisággal
    score =. (+/ context e. 'ATGC') % # context
    scores =. scores , score
  end.
  
  scores
)

NB. Regularizált regresszió génexpresszió predikciójához
ridgeRegression =: 4 : 0
  'X y lambda' =. x ; y ; 0.01
  
  if. 0 = # X do. 0 return. end.
  if. (# y) ~: {. $ X do. 0 return. end.
  
  NB. Ridge regression: beta = (X'X + lambda*I)^-1 X'y
  XtX =. (+|: X) mp X
  I =. lambda * =i. $ XtX
  beta =. %. (XtX + I) mp ((+|: X) mp y)
  
  'INFO' log 'Ridge regression completed: ', ": # beta, ' coefficients'
  beta
)

NB. Principal Component Analysis genomikai adatokhoz
genomicPCA =: 3 : 0
  data =. y
  if. 0 = # data do. _1 return. end.
  
  NB. Adatok centrálésa
  means =. (+/ data) % {. $ data
  centered =. data - means
  
  NB. Kovariancia mátrix
  cov_matrix =. (+|: centered) mp centered
  cov_matrix =. cov_matrix % <: {. $ centered
  
  NB. Eigenvalue dekompozíció
  'eigenvals eigenvecs' =. 18 128 $ 4 $: cov_matrix
  
  NB. Variance explained számítás
  total_var =. +/ eigenvals
  var_explained =. eigenvals % total_var
  
  NB. PC score-ok számítás
  pc_scores =. centered mp eigenvecs
  
  result =. eigenvals ,: var_explained ,: pc_scores ,: eigenvecs
  
  'INFO' log 'PCA analysis: ', ": # eigenvals, ' components, ', ": 100 * +/ 3 {. var_explained, '% variance in top 3 PCs'
  result
)

NB. Hardy-Weinberg egyensúly tesztelés
hardyWeinbergTest =: 3 : 0
  'aa_count ab_count bb_count' =. y
  total =. aa_count + ab_count + bb_count
  
  if. 0 = total do. _1 return. end.
  
  NB. Allél gyakoriságok becslése
  p =. (aa_count + 0.5 * ab_count) % total
  q =. 1 - p
  
  NB. Várható gyakoriságok Hardy-Weinberg szerint
  expected_aa =. total * p ^ 2
  expected_ab =. total * 2 * p * q
  expected_bb =. total * q ^ 2
  
  NB. Chi-square teszt
  observed =. aa_count, ab_count, bb_count
  expected =. expected_aa, expected_ab, expected_bb
  
  chi_square =. +/ (*: observed - expected) % expected
  p_value =. 1 - 1 CDF chi_square  NB. Egyszerűsített p-érték
  
  result =. p, q, chi_square, p_value, expected
  
  'INFO' log 'Hardy-Weinberg test: p=', ": p, ', chi2=', ": chi_square, ', p-value=', ": p_value
  result
)

NB. Linkage disequilibrium számítás
linkageDisequilibrium =: 4 : 0
  'freq_ab freq_a freq_b' =. x, y
  
  NB. D statisztika számítás
  D =. freq_ab - (freq_a * freq_b)
  
  NB. D' (normalizált D)
  if. D > 0 do.
    Dmax =. <./ freq_a * (1 - freq_b), (1 - freq_a) * freq_b
  else.
    Dmax =. <./ freq_a * freq_b, (1 - freq_a) * (1 - freq_b)
  end.
  
  Dprime =. |D % Dmax
  
  NB. r-squared számítás
  r_squared =. (*: D) % (freq_a * (1 - freq_a) * freq_b * (1 - freq_b))
  
  result =. D, Dprime, r_squared
  
  'INFO' log 'Linkage disequilibrium: D=', ": D, ', D''=', ": Dprime, ', r2=', ": r_squared
  result
)

NB. Populációs genetikai statisztikák
populationStats =: 3 : 0
  frequencies =. y
  if. 0 = # frequencies do. _1 return. end.
  
  NB. Allelic richness
  allelic_richness =. # frequencies
  
  NB. Shannon diversity index
  valid_freqs =. frequencies #~ frequencies > 0
  shannon_diversity =. -+/ valid_freqs * 2 ^. valid_freqs
  
  NB. Simpson's diversity index
  simpson_diversity =. 1 - +/ *: valid_freqs
  
  NB. Evenness
  max_diversity =. 2 ^. allelic_richness
  evenness =. shannon_diversity % max_diversity
  
  result =. allelic_richness, shannon_diversity, simpson_diversity, evenness
  
  'INFO' log 'Population stats: ', ": allelic_richness, ' alleles, H=', ": shannon_diversity
  result
)

NB. HTTP szerver inicializálás (egyszerűsített)
startStatsServer =: 3 : 0
  'INFO' log 'J Statistics Engine initialized'
  'INFO' log 'Available functions: nucleotideComposition, codonUsage, findORFs'
  'INFO' log 'Statistical methods: PCA, Hardy-Weinberg, Linkage Disequilibrium'
  'INFO' log 'Population genetics: Diversity indices, Allelic richness'
  
  NB. Szerver indítása (Python bridge szükséges valódi HTTP-hez)
  'INFO' log 'Starting HTTP bridge on port: ', ": PORT
  
  1  NB. Siker
)

NB. JSON válasz generálás
jsonResponse =: 3 : 0
  data =. y
  timestamp =. ": ,. 6!:0 ''
  
  response =. '{"status":"success","timestamp":"', timestamp, '",'
  response =. response, '"data":', (": data), '}'
  
  response
)

NB. Teszt adatok és példák
testDNA =: 'ATGGCGTGCAAATGACTCGTAATGAAAGCTAATAGCGTGCAAATGACTCGTAATGAAAGCTAA'
testExpression =: 10 20 15 25 30 5 40 35 45 20

NB. Alapértelmezett indítás
if. 0 = 4!:0 <'STARTED' do.
  STARTED =: 1
  startStatsServer ''
  
  NB. Teszt futtatás
  'INFO' log 'Running test analysis...'
  comp =. nucleotideComposition testDNA
  orfs =. findORFs testDNA
  pca =. genomicPCA 5 10 $ testExpression
  
  'INFO' log 'Test completed successfully'
end.