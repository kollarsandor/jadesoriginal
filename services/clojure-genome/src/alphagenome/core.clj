(ns alphagenome.core
  "JADED AlphaGenome Core Service (Clojure)
   A genomikai agy - Komplex genomikai adatok elemzése és predikciója"
  (:require [org.httpkit.server :as server]
            [compojure.core :refer [defroutes GET POST]]
            [compojure.route :as route]
            [ring.middleware.json :refer [wrap-json-response wrap-json-body]]
            [ring.middleware.defaults :refer [wrap-defaults api-defaults]]
            [ring.util.response :refer [response status header]]
            [cheshire.core :as json]
            [clojure.tools.logging :as log]
            [clojure.core.async :as async :refer [go <! >! chan]]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [java-time.api :as time]
            [environ.core :refer [env]]
            [alphagenome.bigquery :as bq]
            [alphagenome.genomics :as genomics]
            [alphagenome.ml :as ml]
            [alphagenome.cache :as cache])
  (:gen-class))

(def ^:const PORT 8002)
(def ^:const SERVICE-NAME "AlphaGenome Core (Clojure)")
(def ^:const SERVICE-DESC "A genomikai agy - Funkcionális genomikai elemzés")

;; Service state
(def service-state (atom {:started-at (time/instant)
                         :predictions-count 0
                         :cache-hits 0
                         :bigquery-queries 0}))

;; Health check endpoint
(defn health-check []
  (response {:status "healthy"
             :service SERVICE-NAME
             :description SERVICE-DESC
             :uptime-seconds (-> @service-state :started-at time/duration-between (time/instant) .getSeconds)
             :statistics @service-state
             :timestamp (str (time/instant))}))

;; Service info endpoint
(defn service-info []
  (response {:service_name "AlphaGenome Core"
             :language "Clojure"
             :version "1.0.0"
             :description "Komplex genomikai adatelemzés funkcionális programozási paradigmákkal"
             :features ["BigQuery genomikai adatbázis integráció"
                       "Valós idejű génsebességi predikció"
                       "Tissue-specifikus moduláció (GTEx alapú)"
                       "Regulatory element analízis"
                       "CpG methylation predikció"
                       "ORF és splice site detektálás"
                       "Promoter és enhancer azonosítás"
                       "Funkcionális genomikai pipeline"]
             :capabilities {:max_sequence_length 100000
                           :supported_organisms ["homo_sapiens" "mus_musculus" "danio_rerio" "drosophila_melanogaster"]
                           :supported_tissues (genomics/supported-tissues)
                           :output_formats ["JSON" "CSV" "BigQuery"]
                           :real_time_processing true
                           :bigquery_integration true}}))

;; Genomic sequence analysis endpoint
(defn analyze-sequence [request]
  (try
    (let [body (:body request)
          sequence (:sequence body)
          organism (get body :organism "homo_sapiens")
          tissue (get body :tissue "multi_tissue")
          analysis-type (get body :analysis_type "comprehensive")]
      
      (log/info (format "Analyzing genomic sequence: length=%d, organism=%s, tissue=%s" 
                       (count sequence) organism tissue))
      
      (when (empty? sequence)
        (throw (ex-info "Genomic sequence required" {:status 400})))
      
      (when (> (count sequence) 100000)
        (throw (ex-info "Sequence too long (max 100,000 bp)" {:status 400})))
      
      ;; Check cache first
      (if-let [cached-result (cache/get-analysis sequence organism tissue)]
        (do
          (swap! service-state update :cache-hits inc)
          (-> cached-result
              (assoc :cached true)
              (assoc :timestamp (str (time/instant)))
              response))
        
        ;; Perform comprehensive genomic analysis
        (let [start-time (System/currentTimeMillis)
              
              ;; Core genomic features
              gc-content (genomics/calculate-gc-content sequence)
              cpg-islands (genomics/find-cpg-islands sequence)
              orfs (genomics/find-orfs sequence)
              splice-sites (genomics/predict-splice-sites sequence)
              
              ;; Regulatory elements
              promoters (genomics/predict-promoters sequence organism)
              enhancers (genomics/predict-enhancers sequence organism)
              
              ;; Gene expression prediction using ML models
              expression-prediction (ml/predict-gene-expression sequence organism tissue)
              
              ;; BigQuery integration for population genetics
              population-data (when (genomics/is-known-region? sequence organism)
                               (bq/query-population-genetics sequence organism))
              
              ;; Functional annotations
              functional-annotations (genomics/annotate-functional-elements sequence organism)
              
              processing-time (- (System/currentTimeMillis) start-time)
              
              result {:sequence sequence
                     :length (count sequence)
                     :organism organism
                     :tissue tissue
                     :analysis_type analysis-type
                     :gc_content gc-content
                     :cpg_islands cpg-islands
                     :orfs orfs
                     :splice_sites splice-sites
                     :promoters promoters
                     :enhancers enhancers
                     :expression_prediction expression-prediction
                     :population_genetics population-data
                     :functional_annotations functional-annotations
                     :statistics {:processing_time_ms processing-time
                                 :features_detected (+ (count cpg-islands)
                                                      (count orfs)
                                                      (count splice-sites)
                                                      (count promoters)
                                                      (count enhancers))
                                 :gc_percentage (* gc-content 100.0)
                                 :coding_potential (genomics/calculate-coding-potential orfs)
                                 :regulatory_density (/ (+ (count promoters) (count enhancers))
                                                       (/ (count sequence) 1000.0))}
                     :metadata {:service SERVICE-NAME
                               :timestamp (str (time/instant))
                               :version "AlphaGenome-JADED-v1.0"
                               :cached false}}]
          
          ;; Cache the result
          (cache/store-analysis sequence organism tissue result)
          
          ;; Update service statistics
          (swap! service-state update :predictions-count inc)
          
          (log/info (format "Genomic analysis completed in %dms" processing-time))
          (response result))))
    
    (catch Exception e
      (log/error e "Error during genomic analysis")
      (-> {:error "Internal server error"
           :message (.getMessage e)
           :service SERVICE-NAME}
          response
          (status 500)))))

;; Batch genomic analysis endpoint
(defn batch-analyze [request]
  (try
    (let [body (:body request)
          sequences (:sequences body)
          organism (get body :organism "homo_sapiens")
          tissue (get body :tissue "multi_tissue")]
      
      (when (empty? sequences)
        (throw (ex-info "Sequences required for batch analysis" {:status 400})))
      
      (when (> (count sequences) 20)
        (throw (ex-info "Maximum 20 sequences per batch" {:status 400})))
      
      (log/info (format "Processing batch analysis for %d sequences" (count sequences)))
      
      (let [start-time (System/currentTimeMillis)
            
            ;; Process sequences in parallel using core.async
            results-chan (chan (count sequences))
            
            ;; Submit analysis tasks
            _ (doseq [[idx sequence] (map-indexed vector sequences)]
                (go
                  (try
                    (let [analysis (genomics/quick-analysis sequence organism tissue)]
                      (>! results-chan {:sequence_id idx
                                       :sequence sequence
                                       :analysis analysis
                                       :status :success}))
                    (catch Exception e
                      (>! results-chan {:sequence_id idx
                                       :sequence sequence
                                       :error (.getMessage e)
                                       :status :error})))))
            
            ;; Collect results
            results (loop [collected []
                          remaining (count sequences)]
                     (if (zero? remaining)
                       collected
                       (recur (conj collected (<!! results-chan))
                              (dec remaining))))
            
            total-time (- (System/currentTimeMillis) start-time)
            successful (count (filter #(= :success (:status %)) results))]
        
        (log/info (format "Batch analysis completed: %d/%d successful in %dms" 
                         successful (count sequences) total-time))
        
        (response {:results results
                  :batch_size (count sequences)
                  :successful_analyses successful
                  :failed_analyses (- (count sequences) successful)
                  :total_time_ms total-time
                  :average_time_per_sequence (/ total-time (count sequences))
                  :timestamp (str (time/instant))})))
    
    (catch Exception e
      (log/error e "Error during batch genomic analysis")
      (-> {:error "Internal server error"
           :message (.getMessage e)}
          response
          (status 500)))))

;; Gene expression prediction endpoint
(defn predict-expression [request]
  (try
    (let [body (:body request)
          gene-id (:gene_id body)
          tissue (:tissue body)
          condition (:condition body "normal")]
      
      (when (str/blank? gene-id)
        (throw (ex-info "Gene ID required" {:status 400})))
      
      (log/info (format "Predicting expression for gene %s in tissue %s" gene-id tissue))
      
      (let [prediction (ml/predict-tissue-expression gene-id tissue condition)
            population-data (bq/query-gene-expression-data gene-id)]
        
        (swap! service-state update :bigquery-queries inc)
        
        (response {:gene_id gene-id
                  :tissue tissue
                  :condition condition
                  :predicted_expression prediction
                  :population_statistics population-data
                  :confidence (ml/calculate-expression-confidence prediction population-data)
                  :metadata {:service SERVICE-NAME
                            :timestamp (str (time/instant))
                            :model_version "AlphaGenome-Expression-v2.1"}})))
    
    (catch Exception e
      (log/error e "Error predicting gene expression")
      (-> {:error "Internal server error"
           :message (.getMessage e)}
          response
          (status 500)))))

;; Define routes
(defroutes app-routes
  (GET "/health" [] (health-check))
  (GET "/info" [] (service-info))
  (POST "/analyze" request (analyze-sequence request))
  (POST "/batch_analyze" request (batch-analyze request))
  (POST "/predict_expression" request (predict-expression request))
  (route/not-found {:error "Not Found" :service SERVICE-NAME}))

;; Application with middleware
(def app
  (-> app-routes
      (wrap-json-body {:keywords? true})
      wrap-json-response
      (wrap-defaults api-defaults)))

;; Server management
(def server (atom nil))

(defn start-server []
  (log/info (format "Starting %s on port %d" SERVICE-NAME PORT))
  (reset! server (server/run-server app {:port PORT})))

(defn stop-server []
  (when @server
    (log/info "Stopping AlphaGenome Core service")
    (@server)
    (reset! server nil)))

(defn -main [& args]
  (log/info "Initializing AlphaGenome Core service...")
  
  ;; Initialize components
  (cache/initialize!)
  (bq/initialize!)
  (ml/load-models!)
  
  (log/info "✓ Cache system initialized")
  (log/info "✓ BigQuery connection established")
  (log/info "✓ ML models loaded")
  
  ;; Start HTTP server
  (start-server)
  
  (log/info (format "🧬 %s ready and listening on port %d" SERVICE-NAME PORT))
  (log/info "🔬 A genomikai agy - Funkcionális genomikai elemzés aktiválva")
  
  ;; Shutdown hook
  (.addShutdownHook (Runtime/getRuntime)
                    (Thread. ^Runnable stop-server)))