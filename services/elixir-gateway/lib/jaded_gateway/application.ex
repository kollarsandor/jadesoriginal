defmodule JadedGateway.Application do
  @moduledoc """
  JADED API Gateway Application
  A rendszer "agyja" és "idegrendszere" - coordinates all microservices
  """
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Database
      JadedGateway.Repo,
      
      # PubSub for real-time features
      {Phoenix.PubSub, name: JadedGateway.PubSub},
      
      # Finch HTTP client
      {Finch, name: JadedGateway.Finch},
      
      # Circuit breakers for microservice resilience
      JadedGateway.CircuitBreakers,
      
      # Service registry and health checks
      JadedGateway.ServiceRegistry,
      
      # Rate limiting
      JadedGateway.RateLimiter,
      
      # Background job processing
      {Oban, Application.fetch_env!(:jaded_gateway, Oban)},
      
      # Cache layer
      {JadedGateway.Cache, []},
      
      # Presence tracking
      JadedGateway.Presence,
      
      # Web endpoint
      JadedGatewayWeb.Endpoint,
      
      # Telemetry supervisor
      JadedGateway.Telemetry
    ]

    opts = [strategy: :one_for_one, name: JadedGateway.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @impl true
  def config_change(changed, _new, removed) do
    JadedGatewayWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end