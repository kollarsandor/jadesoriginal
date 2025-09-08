defmodule JadedGateway.ServiceRegistry do
  @moduledoc """
  Service Registry and Health Check Manager
  Manages discovery and health monitoring of all JADED microservices
  """
  use GenServer
  require Logger

  @services %{
    alphafold_core: %{
      name: "AlphaFold 3 Core (Julia)",
      url: "http://alphafold-core:8001",
      health_endpoint: "/health",
      description: "A számítási izom - Nagy teljesítményű protein predikció"
    },
    alphagenome_core: %{
      name: "AlphaGenome Core (Clojure)", 
      url: "http://alphagenome-core:8002",
      health_endpoint: "/health",
      description: "A genomikai agy - Funkcionális genomikai elemzés"
    },
    gcp_client: %{
      name: "GCP Client (Nim)",
      url: "http://gcp-client:8003", 
      health_endpoint: "/health",
      description: "A felhő-kapcsolat - Hatékony felhő adatkezelés"
    },
    federated_learning: %{
      name: "Federated Learning (Pony)",
      url: "http://federated-learning:8004",
      health_endpoint: "/health", 
      description: "A megbízható tanuló - Hibatűrő federált tanulás"
    },
    system_utils: %{
      name: "System Utils (Zig)",
      url: "http://system-utils:8005",
      health_endpoint: "/health",
      description: "A precíziós eszköz - Alacsony szintű optimalizált műveletek"
    },
    logic_engine: %{
      name: "Logic Engine (Prolog)",
      url: "http://logic-engine:8006", 
      health_endpoint: "/health",
      description: "A tudásmotor - Komplex biológiai szabályok és logikai következtetések"
    },
    stats_engine: %{
      name: "Stats Engine (J)",
      url: "http://stats-engine:8007",
      health_endpoint: "/health", 
      description: "A matematikai gyorsító - Ad hoc adatelemzés és statisztika"
    },
    visualization: %{
      name: "Visualization (Pharo)",
      url: "http://visualization:8008",
      health_endpoint: "/health",
      description: "A vizuális labor - Interaktív adatelemzés és vizualizáció"
    },
    protocol_engine: %{
      name: "Protocol Engine (Haskell)", 
      url: "http://protocol-engine:8009",
      health_endpoint: "/health",
      description: "A típusbiztos protokoll - Kritikus protokoll implementációk"
    }
  }

  # Client API
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_services do
    GenServer.call(__MODULE__, :get_services)
  end

  def get_service(service_id) do
    GenServer.call(__MODULE__, {:get_service, service_id})
  end

  def get_healthy_services do
    GenServer.call(__MODULE__, :get_healthy_services)
  end

  def get_service_status(service_id) do
    GenServer.call(__MODULE__, {:get_service_status, service_id})
  end

  def force_health_check do
    GenServer.cast(__MODULE__, :force_health_check)
  end

  # Server callbacks
  @impl true
  def init(_opts) do
    # Initialize service status
    initial_state = 
      @services
      |> Enum.into(%{}, fn {id, config} ->
        {id, Map.put(config, :status, :unknown)}
      end)

    # Schedule periodic health checks
    schedule_health_check()

    {:ok, initial_state}
  end

  @impl true
  def handle_call(:get_services, _from, state) do
    {:reply, state, state}
  end

  @impl true 
  def handle_call({:get_service, service_id}, _from, state) do
    service = Map.get(state, service_id)
    {:reply, service, state}
  end

  @impl true
  def handle_call(:get_healthy_services, _from, state) do
    healthy_services = 
      state
      |> Enum.filter(fn {_id, service} -> service.status == :healthy end)
      |> Enum.into(%{})
    
    {:reply, healthy_services, state}
  end

  @impl true
  def handle_call({:get_service_status, service_id}, _from, state) do
    status = 
      case Map.get(state, service_id) do
        nil -> :not_found
        service -> service.status
      end
    
    {:reply, status, state}
  end

  @impl true
  def handle_cast(:force_health_check, state) do
    new_state = perform_health_checks(state)
    {:noreply, new_state}
  end

  @impl true
  def handle_info(:health_check, state) do
    new_state = perform_health_checks(state)
    schedule_health_check()
    {:noreply, new_state}
  end

  # Private functions
  defp schedule_health_check do
    Process.send_after(self(), :health_check, 30_000) # Every 30 seconds
  end

  defp perform_health_checks(state) do
    Logger.info("Performing health checks on all services")
    
    tasks = 
      state
      |> Enum.map(fn {service_id, service_config} ->
        Task.async(fn -> 
          {service_id, check_service_health(service_config)}
        end)
      end)

    # Wait for all health checks to complete (with timeout)
    results = 
      tasks
      |> Task.await_many(10_000) # 10 second timeout per service
      |> Enum.into(%{})

    # Update service statuses
    updated_state = 
      state
      |> Enum.into(%{}, fn {service_id, service_config} ->
        new_status = Map.get(results, service_id, :timeout)
        updated_config = Map.put(service_config, :status, new_status)
        updated_config = Map.put(updated_config, :last_check, DateTime.utc_now())
        {service_id, updated_config}
      end)

    # Log status changes
    log_status_changes(state, updated_state)

    updated_state
  end

  defp check_service_health(service_config) do
    health_url = service_config.url <> service_config.health_endpoint
    
    case Req.get(health_url, connect_options: [timeout: 5000]) do
      {:ok, %{status: 200}} -> 
        :healthy
      {:ok, %{status: status}} -> 
        Logger.warn("Service #{service_config.name} returned status #{status}")
        :unhealthy
      {:error, reason} ->
        Logger.warn("Service #{service_config.name} health check failed: #{inspect(reason)}")
        :unhealthy
    end
  rescue
    e ->
      Logger.warn("Health check exception for #{service_config.name}: #{inspect(e)}")
      :unhealthy
  end

  defp log_status_changes(old_state, new_state) do
    Enum.each(new_state, fn {service_id, new_service} ->
      old_status = get_in(old_state, [service_id, :status])
      new_status = new_service.status
      
      if old_status != new_status do
        Logger.info("Service #{new_service.name} status changed: #{old_status} -> #{new_status}")
        
        # Publish status change event
        Phoenix.PubSub.broadcast(
          JadedGateway.PubSub,
          "service_status",
          {:service_status_change, service_id, old_status, new_status}
        )
      end
    end)
  end
end