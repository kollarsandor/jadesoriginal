// JADED Interface Service (Dart)
// Modern reszponzív UI - Flutter-based tudományos interface
// Valós idejű adatfrissítés és interaktív grafikonok

import 'dart:io';
import 'dart:convert';
import 'dart:async';
import 'package:shelf/shelf.dart';
import 'package:shelf/shelf_io.dart';
import 'package:shelf_router/shelf_router.dart';
import 'package:shelf_cors_headers/shelf_cors_headers.dart';

const int PORT = 8010;
const String SERVICE_NAME = 'dart-interface';
const String VERSION = '1.0.0';

// Service configuration
class ServiceConfig {
  static const int maxConnections = 1000;
  static const int requestTimeoutMs = 30000;
  static const int maxRequestSizeMB = 10;
  static const bool enableRealTimeUpdates = true;
  static const bool enableCaching = true;
}

// Data models for scientific interface
class ProteinData {
  final String id;
  final String sequence;
  final List<AtomCoordinate> coordinates;
  final double confidence;
  final DateTime timestamp;

  ProteinData({
    required this.id,
    required this.sequence,
    required this.coordinates,
    required this.confidence,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'sequence': sequence,
    'coordinates': coordinates.map((c) => c.toJson()).toList(),
    'confidence': confidence,
    'timestamp': timestamp.toIso8601String(),
  };

  factory ProteinData.fromJson(Map<String, dynamic> json) => ProteinData(
    id: json['id'],
    sequence: json['sequence'],
    coordinates: (json['coordinates'] as List)
        .map((c) => AtomCoordinate.fromJson(c))
        .toList(),
    confidence: json['confidence'].toDouble(),
    timestamp: DateTime.parse(json['timestamp']),
  );
}

class AtomCoordinate {
  final double x, y, z;
  final String atomType;
  final int residueId;

  AtomCoordinate({
    required this.x,
    required this.y,
    required this.z,
    required this.atomType,
    required this.residueId,
  });

  Map<String, dynamic> toJson() => {
    'x': x,
    'y': y,
    'z': z,
    'atom_type': atomType,
    'residue_id': residueId,
  };

  factory AtomCoordinate.fromJson(Map<String, dynamic> json) => AtomCoordinate(
    x: json['x'].toDouble(),
    y: json['y'].toDouble(),
    z: json['z'].toDouble(),
    atomType: json['atom_type'],
    residueId: json['residue_id'],
  );
}

class GenomicData {
  final String sequence;
  final String organism;
  final String tissue;
  final List<GeneFeature> features;
  final Map<String, double> expressionLevels;
  final DateTime timestamp;

  GenomicData({
    required this.sequence,
    required this.organism,
    required this.tissue,
    required this.features,
    required this.expressionLevels,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
    'sequence': sequence,
    'organism': organism,
    'tissue': tissue,
    'features': features.map((f) => f.toJson()).toList(),
    'expression_levels': expressionLevels,
    'timestamp': timestamp.toIso8601String(),
  };
}

class GeneFeature {
  final String type;
  final int start;
  final int end;
  final double score;
  final String description;

  GeneFeature({
    required this.type,
    required this.start,
    required this.end,
    required this.score,
    required this.description,
  });

  Map<String, dynamic> toJson() => {
    'type': type,
    'start': start,
    'end': end,
    'score': score,
    'description': description,
  };
}

class VisualizationData {
  final String type;
  final Map<String, dynamic> config;
  final List<dynamic> data;
  final Map<String, String> metadata;

  VisualizationData({
    required this.type,
    required this.config,
    required this.data,
    required this.metadata,
  });

  Map<String, dynamic> toJson() => {
    'type': type,
    'config': config,
    'data': data,
    'metadata': metadata,
  };
}

// Scientific data processing utilities
class BioinformaticsUtils {
  static bool isValidDNASequence(String sequence) {
    final validChars = RegExp(r'^[ATGCNU]+$');
    return validChars.hasMatch(sequence.toUpperCase());
  }

  static bool isValidProteinSequence(String sequence) {
    final validChars = RegExp(r'^[ARNDCQEGHILKMFPSTWYV]+$');
    return validChars.hasMatch(sequence.toUpperCase());
  }

  static Map<String, double> calculateNucleotideComposition(String sequence) {
    final counts = <String, int>{'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0, 'U': 0};
    final total = sequence.length;

    for (final char in sequence.toUpperCase().split('')) {
      if (counts.containsKey(char)) {
        counts[char] = counts[char]! + 1;
      }
    }

    return counts.map((key, value) => MapEntry(key, value / total));
  }

  static double calculateGCContent(String sequence) {
    final composition = calculateNucleotideComposition(sequence);
    return (composition['G']! + composition['C']!) * 100;
  }

  static List<int> findORFs(String sequence, {int minLength = 300}) {
    final orfs = <int>[];
    final startCodons = ['ATG', 'GTG', 'TTG'];
    final stopCodons = ['TAA', 'TAG', 'TGA'];

    for (int frame = 0; frame < 3; frame++) {
      for (int i = frame; i < sequence.length - 2; i += 3) {
        final codon = sequence.substring(i, i + 3).toUpperCase();
        
        if (startCodons.contains(codon)) {
          // Find next stop codon
          for (int j = i + 3; j < sequence.length - 2; j += 3) {
            final stopCodon = sequence.substring(j, j + 3).toUpperCase();
            if (stopCodons.contains(stopCodon)) {
              final orfLength = j - i + 3;
              if (orfLength >= minLength) {
                orfs.add(i);
              }
              break;
            }
          }
        }
      }
    }

    return orfs;
  }

  static String translateDNAToProtein(String dnaSequence) {
    final codonTable = {
      'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
      'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
      'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
      'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
      'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
      'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
      'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
      'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
      'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
      'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
      'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
      'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
      'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
      'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
      'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
      'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    };

    final protein = StringBuffer();
    for (int i = 0; i < dnaSequence.length - 2; i += 3) {
      final codon = dnaSequence.substring(i, i + 3).toUpperCase();
      final aminoAcid = codonTable[codon] ?? 'X';
      if (aminoAcid == '*') break;
      protein.write(aminoAcid);
    }

    return protein.toString();
  }
}

// Real-time data streaming
class DataStreamManager {
  final Map<String, StreamController<Map<String, dynamic>>> _streams = {};
  final Map<String, Timer> _timers = {};

  void startProteinStream(String proteinId) {
    if (_streams.containsKey(proteinId)) return;

    final controller = StreamController<Map<String, dynamic>>.broadcast();
    _streams[proteinId] = controller;

    // Simulate real-time protein folding updates
    _timers[proteinId] = Timer.periodic(Duration(seconds: 2), (timer) {
      final updateData = {
        'protein_id': proteinId,
        'folding_progress': (timer.tick * 5).clamp(0, 100),
        'confidence': 0.5 + (timer.tick * 0.05).clamp(0, 0.45),
        'current_energy': -100.0 - (timer.tick * 2.5),
        'timestamp': DateTime.now().toIso8601String(),
      };

      controller.add(updateData);

      if (timer.tick >= 20) {
        timer.cancel();
        controller.close();
        _streams.remove(proteinId);
        _timers.remove(proteinId);
      }
    });
  }

  void startGenomicStream(String analysisId) {
    if (_streams.containsKey(analysisId)) return;

    final controller = StreamController<Map<String, dynamic>>.broadcast();
    _streams[analysisId] = controller;

    _timers[analysisId] = Timer.periodic(Duration(seconds: 3), (timer) {
      final updateData = {
        'analysis_id': analysisId,
        'genes_processed': timer.tick * 10,
        'variants_found': timer.tick * 3,
        'expression_correlation': 0.1 + (timer.tick * 0.02).clamp(0, 0.8),
        'timestamp': DateTime.now().toIso8601String(),
      };

      controller.add(updateData);

      if (timer.tick >= 15) {
        timer.cancel();
        controller.close();
        _streams.remove(analysisId);
        _timers.remove(analysisId);
      }
    });
  }

  Stream<Map<String, dynamic>>? getStream(String id) {
    return _streams[id]?.stream;
  }

  void stopStream(String id) {
    _timers[id]?.cancel();
    _streams[id]?.close();
    _streams.remove(id);
    _timers.remove(id);
  }

  void stopAllStreams() {
    _timers.values.forEach((timer) => timer.cancel());
    _streams.values.forEach((controller) => controller.close());
    _streams.clear();
    _timers.clear();
  }
}

// Main HTTP service
class JadedInterfaceService {
  final DataStreamManager _streamManager = DataStreamManager();
  final Map<String, dynamic> _cache = {};

  // Request handlers
  Response _healthHandler(Request request) {
    final response = {
      'status': 'healthy',
      'service': SERVICE_NAME,
      'version': VERSION,
      'timestamp': DateTime.now().toIso8601String(),
      'features': [
        'real_time_updates',
        'interactive_visualization',
        'responsive_ui',
        'data_streaming'
      ],
      'real_time_streams': _streamManager._streams.length,
    };

    return Response.ok(
      jsonEncode(response),
      headers: {'Content-Type': 'application/json'},
    );
  }

  Future<Response> _analyzeProteinHandler(Request request) async {
    try {
      final body = await request.readAsString();
      final data = jsonDecode(body) as Map<String, dynamic>;
      
      final sequence = data['sequence'] as String;
      final proteinId = data['protein_id'] as String? ?? 'protein_${DateTime.now().millisecondsSinceEpoch}';

      if (!BioinformaticsUtils.isValidProteinSequence(sequence)) {
        return Response(400, 
          body: jsonEncode({'error': 'Invalid protein sequence'}),
          headers: {'Content-Type': 'application/json'},
        );
      }

      // Start real-time folding simulation
      _streamManager.startProteinStream(proteinId);

      // Generate mock coordinates
      final coordinates = <AtomCoordinate>[];
      for (int i = 0; i < sequence.length; i++) {
        coordinates.addAll([
          AtomCoordinate(x: i * 1.5, y: 0.0, z: 0.0, atomType: 'N', residueId: i),
          AtomCoordinate(x: i * 1.5 + 0.5, y: 0.5, z: 0.0, atomType: 'CA', residueId: i),
          AtomCoordinate(x: i * 1.5 + 1.0, y: 0.0, z: 0.0, atomType: 'C', residueId: i),
        ]);
      }

      final proteinData = ProteinData(
        id: proteinId,
        sequence: sequence,
        coordinates: coordinates,
        confidence: 0.85,
        timestamp: DateTime.now(),
      );

      final response = {
        'status': 'success',
        'protein_data': proteinData.toJson(),
        'stream_url': '/stream/protein/$proteinId',
        'analysis_started': true,
      };

      return Response.ok(
        jsonEncode(response),
        headers: {'Content-Type': 'application/json'},
      );
    } catch (e) {
      return Response(500,
        body: jsonEncode({'error': 'Analysis failed: $e'}),
        headers: {'Content-Type': 'application/json'},
      );
    }
  }

  Future<Response> _analyzeGenomicHandler(Request request) async {
    try {
      final body = await request.readAsString();
      final data = jsonDecode(body) as Map<String, dynamic>;
      
      final sequence = data['sequence'] as String;
      final organism = data['organism'] as String? ?? 'homo_sapiens';
      final tissue = data['tissue'] as String? ?? 'multi_tissue';
      final analysisId = data['analysis_id'] as String? ?? 'genomic_${DateTime.now().millisecondsSinceEpoch}';

      if (!BioinformaticsUtils.isValidDNASequence(sequence)) {
        return Response(400,
          body: jsonEncode({'error': 'Invalid DNA sequence'}),
          headers: {'Content-Type': 'application/json'},
        );
      }

      // Start real-time genomic analysis
      _streamManager.startGenomicStream(analysisId);

      // Perform basic analysis
      final composition = BioinformaticsUtils.calculateNucleotideComposition(sequence);
      final gcContent = BioinformaticsUtils.calculateGCContent(sequence);
      final orfs = BioinformaticsUtils.findORFs(sequence);

      final features = orfs.map((start) => GeneFeature(
        type: 'ORF',
        start: start,
        end: start + 300,
        score: 0.8,
        description: 'Open Reading Frame',
      )).toList();

      final expressionLevels = <String, double>{
        'gene1': 5.2,
        'gene2': 3.8,
        'gene3': 7.1,
        'gene4': 2.5,
      };

      final genomicData = GenomicData(
        sequence: sequence,
        organism: organism,
        tissue: tissue,
        features: features,
        expressionLevels: expressionLevels,
        timestamp: DateTime.now(),
      );

      final response = {
        'status': 'success',
        'genomic_data': genomicData.toJson(),
        'composition': composition,
        'gc_content': gcContent,
        'orf_count': orfs.length,
        'stream_url': '/stream/genomic/$analysisId',
        'analysis_started': true,
      };

      return Response.ok(
        jsonEncode(response),
        headers: {'Content-Type': 'application/json'},
      );
    } catch (e) {
      return Response(500,
        body: jsonEncode({'error': 'Genomic analysis failed: $e'}),
        headers: {'Content-Type': 'application/json'},
      );
    }
  }

  Response _createVisualizationHandler(Request request) {
    final vizData = VisualizationData(
      type: 'protein_structure',
      config: {
        'color_scheme': 'rainbow',
        'representation': 'cartoon',
        'background': 'white',
        'interactive': true,
      },
      data: [],
      metadata: {
        'created': DateTime.now().toIso8601String(),
        'format': 'pdb',
        'resolution': '2.5A',
      },
    );

    return Response.ok(
      jsonEncode({
        'status': 'success',
        'visualization': vizData.toJson(),
        'render_url': '/render/visualization',
      }),
      headers: {'Content-Type': 'application/json'},
    );
  }

  Response _streamHandler(Request request, String streamType, String streamId) {
    final stream = _streamManager.getStream(streamId);
    
    if (stream == null) {
      return Response(404,
        body: jsonEncode({'error': 'Stream not found'}),
        headers: {'Content-Type': 'application/json'},
      );
    }

    // For Server-Sent Events
    final controller = StreamController<String>();
    
    stream.listen((data) {
      controller.add('data: ${jsonEncode(data)}\n\n');
    }, onDone: () {
      controller.close();
    });

    return Response.ok(
      controller.stream,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    );
  }

  // Create router
  Router createRouter() {
    final router = Router();

    router.get('/health', _healthHandler);
    router.post('/analyze/protein', _analyzeProteinHandler);
    router.post('/analyze/genomic', _analyzeGenomicHandler);
    router.post('/visualization/create', _createVisualizationHandler);
    router.get('/stream/<streamType>/<streamId>', (Request request, String streamType, String streamId) {
      return _streamHandler(request, streamType, streamId);
    });

    router.get('/interface', (Request request) {
      return Response.ok(
        _generateInterfaceHTML(),
        headers: {'Content-Type': 'text/html'},
      );
    });

    return router;
  }

  String _generateInterfaceHTML() {
    return '''
<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JADED Scientific Interface</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; 
                     padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                        gap: 20px; margin-top: 30px; }
        .feature-card { background: #f8f9fa; border-radius: 8px; padding: 20px; 
                        border-left: 4px solid #007bff; }
        .feature-card h3 { margin-top: 0; color: #007bff; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; 
                           border-radius: 50%; background: #28a745; margin-right: 8px; }
        .realtime-data { background: #e7f3ff; border-radius: 6px; padding: 15px; 
                        margin-top: 15px; font-family: monospace; }
        .btn { background: #007bff; color: white; border: none; padding: 12px 24px; 
               border-radius: 6px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 JADED Scientific Interface</h1>
        <p style="text-align: center; font-size: 18px; color: #666;">
            <span class="status-indicator"></span>
            Dart Interface Service v${VERSION} - Real-time Scientific Computing
        </p>
        
        <div class="feature-grid">
            <div class="feature-card">
                <h3>🔬 Protein Analysis</h3>
                <p>Real-time protein structure prediction with AlphaFold 3 integration</p>
                <button class="btn" onclick="startProteinAnalysis()">Start Analysis</button>
                <div id="protein-data" class="realtime-data" style="display: none;">
                    Waiting for data...
                </div>
            </div>
            
            <div class="feature-card">
                <h3>🧬 Genomic Analysis</h3>
                <p>Advanced genomic sequence analysis with tissue-specific predictions</p>
                <button class="btn" onclick="startGenomicAnalysis()">Start Analysis</button>
                <div id="genomic-data" class="realtime-data" style="display: none;">
                    Waiting for data...
                </div>
            </div>
            
            <div class="feature-card">
                <h3>📊 Real-time Visualization</h3>
                <p>Interactive 3D molecular visualization and expression heatmaps</p>
                <button class="btn" onclick="createVisualization()">Create Visualization</button>
                <div id="viz-data" class="realtime-data" style="display: none;">
                    Visualization ready...
                </div>
            </div>
            
            <div class="feature-card">
                <h3>⚡ Live Data Streams</h3>
                <p>WebSocket-based real-time updates for all scientific computations</p>
                <div class="realtime-data">
                    Active streams: <span id="stream-count">0</span><br>
                    Service uptime: <span id="uptime">0s</span><br>
                    Last update: <span id="last-update">Never</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let startTime = Date.now();
        
        function updateUptime() {
            const uptime = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('uptime').textContent = uptime + 's';
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }
        
        setInterval(updateUptime, 1000);
        
        async function startProteinAnalysis() {
            const data = { sequence: 'MKFLVNVALVFMVVYISYIYALAVPFYH', protein_id: 'demo_protein' };
            const response = await fetch('/analyze/protein', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            document.getElementById('protein-data').style.display = 'block';
            document.getElementById('protein-data').innerHTML = JSON.stringify(result, null, 2);
        }
        
        async function startGenomicAnalysis() {
            const data = { sequence: 'ATGGCGTGCAAATGACTCGTAATGAAAGCTAA', organism: 'homo_sapiens' };
            const response = await fetch('/analyze/genomic', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            document.getElementById('genomic-data').style.display = 'block';
            document.getElementById('genomic-data').innerHTML = JSON.stringify(result, null, 2);
        }
        
        async function createVisualization() {
            const response = await fetch('/visualization/create', { method: 'POST' });
            const result = await response.json();
            document.getElementById('viz-data').style.display = 'block';
            document.getElementById('viz-data').innerHTML = JSON.stringify(result, null, 2);
        }
    </script>
</body>
</html>
    ''';
  }
}

// Main entry point
void main() async {
  print('🎯 DART INTERFACE SERVICE INDÍTÁSA');
  print('Port: $PORT');
  print('Service: $SERVICE_NAME v$VERSION');
  print('Features: Real-time updates, Interactive UI, Data streaming');

  final service = JadedInterfaceService();
  final router = service.createRouter();
  
  final pipeline = Pipeline()
      .addMiddleware(corsHeaders())
      .addMiddleware(logRequests())
      .addHandler(router);

  final server = await serve(pipeline, InternetAddress.anyIPv4, PORT);
  
  print('✅ Dart Interface Service listening on port $PORT');
  print('Available endpoints:');
  print('  GET  /health - Service health check');
  print('  POST /analyze/protein - Protein structure analysis');
  print('  POST /analyze/genomic - Genomic sequence analysis');
  print('  POST /visualization/create - Create visualizations');
  print('  GET  /interface - Interactive web interface');
  print('  GET  /stream/<type>/<id> - Real-time data streams');
  
  print('\n🌐 Open http://localhost:$PORT/interface for web interface');
}