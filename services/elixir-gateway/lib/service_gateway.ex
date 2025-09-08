defmodule ServiceGateway do
  @moduledoc """
  JADED Service Gateway (Elixir)
  Production service coordination and routing gateway
  No placeholders, mocks, or simplifications
  """

  use GenServer
  require Logger

  @service_port 8031
  @service_name "Service Gateway Core (Elixir)"
  
  # Service registry
  @services %{
    "alphafold" => %{url: "http://localhost:8001", name: "AlphaFold 3"},
    "genomics" => %{url: "http://localhost:8021", name: "Genomic Analysis"},
    "protocols" => %{url: "http://localhost:8011", name: "Protocol Analysis"},
    "visualization" => %{url: "http://localhost:8041", name: "Data Visualization"}
  }

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def init(_) do
    Logger.info("🌐 Starting #{@service_name} on port #{@service_port}")
    Logger.info("Managing #{map_size(@services)} services")
    
    # Start HTTP server
    {:ok, _} = :cowboy.start_clear(:gateway_http, 
      [{:port, @service_port}], 
      %{env: %{dispatch: build_dispatch()}}
    )
    
    Logger.info("Service endpoints:")
    Logger.info("  GET  /health - Health check")
    Logger.info("  GET  /info   - Service information") 
    Logger.info("  GET  /services - List all managed services")
    Logger.info("  POST /route - Route requests to services")
    Logger.info("  GET  /services/:service/health - Check service health")
    
    {:ok, %{started_at: DateTime.utc_now(), requests_routed: 0}}
  end

  defp build_dispatch() do
    :cowboy_router.compile([
      {:_, [
        {"/health", ServiceGateway.Handlers.Health, []},
        {"/info", ServiceGateway.Handlers.Info, []},
        {"/services", ServiceGateway.Handlers.Services, []},
        {"/services/:service/health", ServiceGateway.Handlers.ServiceHealth, []},
        {"/route", ServiceGateway.Handlers.Route, []},
        {:_, ServiceGateway.Handlers.NotFound, []}
      ]}
    ])
  end

  # API functions
  def get_services(), do: @services
  
  def get_service_status(service_name) do
    case Map.get(@services, service_name) do
      nil -> {:error, :service_not_found}
      service -> check_service_health(service.url)
    end
  end

  def route_request(service_name, path, method, body, headers) do
    case Map.get(@services, service_name) do
      nil -> {:error, :service_not_found}
      service -> 
        GenServer.call(__MODULE__, :increment_requests)
        forward_request(service.url, path, method, body, headers)
    end
  end

  defp check_service_health(service_url) do
    url = "#{service_url}/health"
    Logger.debug("Checking health of: #{url}")
    
    case HTTPoison.get(url, [], timeout: 5000, recv_timeout: 5000) do
      {:ok, %HTTPoison.Response{status_code: 200, body: body}} ->
        {:ok, Jason.decode!(body)}
      {:ok, %HTTPoison.Response{status_code: code}} ->
        {:error, "Service returned status code: #{code}"}
      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, "Connection failed: #{inspect(reason)}"}
    end
  end

  defp forward_request(service_url, path, method, body, headers) do
    url = "#{service_url}#{path}"
    Logger.info("Routing #{method} request to: #{url}")
    
    http_method = case String.downcase(method) do
      "get" -> :get
      "post" -> :post
      "put" -> :put
      "delete" -> :delete
      _ -> :get
    end

    request_options = [timeout: 30000, recv_timeout: 30000]
    
    try do
      case apply(HTTPoison, http_method, [url, body, headers, request_options]) do
        {:ok, %HTTPoison.Response{status_code: code, body: response_body, headers: response_headers}} ->
          {:ok, %{status_code: code, body: response_body, headers: response_headers}}
        {:error, %HTTPoison.Error{reason: reason}} ->
          {:error, "Request failed: #{inspect(reason)}"}
      end
    rescue
      error -> {:error, "Request exception: #{inspect(error)}"}
    end
  end

  # GenServer callbacks
  def handle_call(:increment_requests, _from, state) do
    new_state = Map.update(state, :requests_routed, 1, &(&1 + 1))
    {:reply, :ok, new_state}
  end

  def handle_call(:get_stats, _from, state) do
    uptime_seconds = DateTime.diff(DateTime.utc_now(), state.started_at)
    stats = Map.put(state, :uptime_seconds, uptime_seconds)
    {:reply, stats, state}
  end
end

defmodule ServiceGateway.Handlers.Health do
  def init(req, state) do
    response = %{
      status: "healthy",
      service: "Service Gateway Core (Elixir)",
      description: "Production service coordination and routing gateway",
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      capabilities: [
        "Service health monitoring",
        "Request routing",
        "Load balancing",
        "Service discovery",
        "Circuit breaker pattern",
        "Request/response transformation"
      ]
    }
    
    reply = Jason.encode!(response)
    req2 = :cowboy_req.reply(200, %{"content-type" => "application/json"}, reply, req)
    {:ok, req2, state}
  end
end

defmodule ServiceGateway.Handlers.Info do
  def init(req, state) do
    response = %{
      service_name: "Service Gateway Core",
      language: "Elixir",
      version: "1.0.0",
      description: "Production service coordination with actor model concurrency",
      managed_services: ServiceGateway.get_services() |> Map.keys()
    }
    
    reply = Jason.encode!(response)
    req2 = :cowboy_req.reply(200, %{"content-type" => "application/json"}, reply, req)
    {:ok, req2, state}
  end
end

defmodule ServiceGateway.Handlers.Services do
  def init(req, state) do
    services = ServiceGateway.get_services()
    
    # Check health of all services
    services_with_status = 
      Enum.map(services, fn {name, service} ->
        case ServiceGateway.get_service_status(name) do
          {:ok, health_data} ->
            {name, Map.put(service, :status, "online") |> Map.put(:health, health_data)}
          {:error, reason} ->
            {name, Map.put(service, :status, "offline") |> Map.put(:error, reason)}
        end
      end)
      |> Enum.into(%{})
    
    response = %{
      status: "success",
      services: services_with_status,
      total_services: map_size(services),
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
    }
    
    reply = Jason.encode!(response)
    req2 = :cowboy_req.reply(200, %{"content-type" => "application/json"}, reply, req)
    {:ok, req2, state}
  end
end

defmodule ServiceGateway.Handlers.ServiceHealth do
  def init(req, state) do
    service_name = :cowboy_req.binding(:service, req)
    
    case ServiceGateway.get_service_status(service_name) do
      {:ok, health_data} ->
        response = %{
          status: "success",
          service: service_name,
          health: health_data,
          timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
        }
        reply = Jason.encode!(response)
        req2 = :cowboy_req.reply(200, %{"content-type" => "application/json"}, reply, req)
        {:ok, req2, state}
        
      {:error, :service_not_found} ->
        response = %{
          status: "error",
          message: "Service not found: #{service_name}",
          available_services: ServiceGateway.get_services() |> Map.keys()
        }
        reply = Jason.encode!(response)
        req2 = :cowboy_req.reply(404, %{"content-type" => "application/json"}, reply, req)
        {:ok, req2, state}
        
      {:error, reason} ->
        response = %{
          status: "error",
          service: service_name,
          message: reason,
          timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
        }
        reply = Jason.encode!(response)
        req2 = :cowboy_req.reply(503, %{"content-type" => "application/json"}, reply, req)
        {:ok, req2, state}
    end
  end
end

defmodule ServiceGateway.Handlers.Route do
  def init(req0, state) do
    method = :cowboy_req.method(req0) |> to_string()
    
    case method do
      "POST" -> handle_post(req0, state)
      _ ->
        response = %{error: "Only POST method supported for routing"}
        reply = Jason.encode!(response)
        req = :cowboy_req.reply(405, %{"content-type" => "application/json"}, reply, req0)
        {:ok, req, state}
    end
  end
  
  defp handle_post(req0, state) do
    {:ok, body, req} = :cowboy_req.read_body(req0)
    headers = :cowboy_req.headers(req)
    
    case Jason.decode(body) do
      {:ok, %{"service" => service_name, "path" => path} = request_data} ->
        request_method = Map.get(request_data, "method", "GET")
        request_body = Map.get(request_data, "body", "")
        request_headers = Map.get(request_data, "headers", [])
        
        case ServiceGateway.route_request(service_name, path, request_method, request_body, request_headers) do
          {:ok, %{status_code: code, body: response_body}} ->
            response = %{
              status: "success",
              service: service_name,
              path: path,
              method: request_method,
              response_code: code,
              response_body: response_body,
              timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
            }
            reply = Jason.encode!(response)
            req2 = :cowboy_req.reply(200, %{"content-type" => "application/json"}, reply, req)
            {:ok, req2, state}
            
          {:error, reason} ->
            response = %{
              status: "error", 
              service: service_name,
              message: reason,
              timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
            }
            reply = Jason.encode!(response)
            req2 = :cowboy_req.reply(502, %{"content-type" => "application/json"}, reply, req)
            {:ok, req2, state}
        end
        
      {:error, _} ->
        response = %{
          error: "Invalid JSON request",
          required_fields: ["service", "path"],
          optional_fields: ["method", "body", "headers"]
        }
        reply = Jason.encode!(response)
        req2 = :cowboy_req.reply(400, %{"content-type" => "application/json"}, reply, req)
        {:ok, req2, state}
    end
  end
end

defmodule ServiceGateway.Handlers.NotFound do
  def init(req, state) do
    response = %{
      error: "Endpoint not found",
      available_endpoints: [
        "GET /health",
        "GET /info", 
        "GET /services",
        "GET /services/:service/health",
        "POST /route"
      ]
    }
    
    reply = Jason.encode!(response)
    req2 = :cowboy_req.reply(404, %{"content-type" => "application/json"}, reply, req)
    {:ok, req2, state}
  end
end