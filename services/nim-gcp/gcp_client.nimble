# Package
version       = "1.0.0"
author        = "JADED Platform"
description   = "A felhő-kapcsolat - Hatékony GCP adatkezelés"
license       = "MIT"
srcDir        = "src"
bin           = @["gcp_service"]

# Dependencies
requires "nim >= 2.0.0"
requires "asynchttpserver >= 0.7.0"
requires "json >= 0.21.0"
requires "httpcore >= 0.4.0"
requires "uri3 >= 0.1.0"
requires "std/asyncnet"
requires "bearssl"
requires "zippy"
requires "chronicles"  # Advanced logging
requires "chronos >= 3.0.0"  # Async framework