defmodule ServiceGateway.Application do
  @moduledoc """
  Application module for the JADED Service Gateway
  """

  use Application
  require Logger

  def start(_type, _args) do
    Logger.info("🚀 JADED Service Gateway Application starting...")
    
    children = [
      ServiceGateway
    ]

    opts = [strategy: :one_for_one, name: ServiceGateway.Supervisor]
    Supervisor.start_link(children, opts)
  end
end