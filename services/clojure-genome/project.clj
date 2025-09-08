(defproject alphagenome "1.0.0"
  :description "JADED AlphaGenome Production Service"
  :url "https://jaded-platform.com"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"}
  
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [ring/ring-core "1.9.6"]
                 [ring/ring-jetty-adapter "1.9.6"]
                 [ring/ring-json "0.5.1"]
                 [ring-cors/ring-cors "0.1.13"]
                 [compojure "1.7.0"]
                 [cheshire "5.11.0"]
                 [org.clojure/data.json "2.4.0"]]
  
  :main genomic-service
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all
                       :jvm-opts ["-Dclojure.compiler.direct-linking=true"]}})