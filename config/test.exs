# Test configuration for CrucibleKitchen
#
# CrucibleKitchen is a library that doesn't require database access.
# These settings are now defaults, but explicit is clearer for test config.
import Config

# All crucible libraries now default to start_repo: false
# Host apps configure: `config :crucible_*, repo: MyApp.Repo`
config :crucible_framework, start_repo: false
config :crucible_model_registry, start_repo: false
config :crucible_feedback, start_repo: false, start_ingestion: false
