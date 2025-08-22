#!/bin/bash
# Install SQLite Read-Only MCP Server for ML Agents Database
# This script installs the SQLite Explorer FastMCP server for read-only access
# to the ml_agents_results.db database within Claude Code.

set -e  # Exit on any error

# Configuration variables
MCP_SERVERS_DIR="${HOME}/.mcp/servers"
SERVER_NAME="sqlite-explorer-fastmcp-mcp-server"
SERVER_REPO="https://github.com/hannesrudolph/sqlite-explorer-fastmcp-mcp-server.git"
MCP_CONFIG_NAME="sqlite-read-only"
PYTHON_VERSION="3.11"
DB_PATH="./ml_agents_results.db"

# Get the current directory (should be the ML Agents project root)
PROJECT_DIR="$(pwd)"

# Check if we're in the ML Agents project root directory
if [ ! -f "pyproject.toml" ] || [ ! -f "CLAUDE.md" ] || [ ! -d "src" ]; then
    echo "❌ Error: This script must be run from the ML Agents project root directory"
    echo "📍 Current directory: ${PROJECT_DIR}"
    echo "🔍 Expected files: pyproject.toml, CLAUDE.md, src/"
    echo "💡 Please cd to the ML Agents project root and run the script again"
    exit 1
fi

echo "🔧 Installing SQLite Read-Only MCP Server for ML Agents..."
echo "📍 Project directory: ${PROJECT_DIR}"
echo "📍 MCP servers directory: ${MCP_SERVERS_DIR}"

# 1. Create MCP servers directory
echo "📁 Creating MCP servers directory..."
mkdir -p "${MCP_SERVERS_DIR}"

# 2. Clone the SQLite Explorer FastMCP server
echo "📦 Cloning SQLite Explorer FastMCP server..."
SERVER_PATH="${MCP_SERVERS_DIR}/${SERVER_NAME}"
if [ -d "${SERVER_PATH}" ]; then
    echo "⚠️  Server directory already exists. Pulling latest changes..."
    cd "${SERVER_PATH}"
    git pull
else
    git clone "${SERVER_REPO}" "${SERVER_PATH}"
fi

# 3. Navigate back to ML Agents project
echo "📂 Returning to ML Agents project..."
cd "${PROJECT_DIR}"

# 4. Add MCP server with project scope
echo "⚙️  Configuring MCP server with project scope..."
claude mcp add "${MCP_CONFIG_NAME}" \
    --scope project \
    --env SQLITE_DB_PATH="${DB_PATH}" \
    -- uvx run --python="${PYTHON_VERSION}" --from="${SERVER_PATH}" python sqlite_explorer.py

echo "✅ Installation complete!"
echo ""
echo "📋 Verification steps:"
echo "1. Run: claude mcp get ${MCP_CONFIG_NAME}"
echo "2. Check project config: cat .mcp.json"
echo "3. In Claude Code, test with: /mcp"
echo ""
echo "ℹ️  Note: 'claude mcp list' won't show project-scoped servers (known bug #5963)"
echo "   Use 'claude mcp get ${MCP_CONFIG_NAME}' to verify installation instead."
echo ""
echo "📊 Database access:"
echo "- Database: ${DB_PATH} (relative to project)"
echo "- Access: Read-only (SELECT queries only)"
echo "- Scope: Project (shared via .mcp.json)"
echo ""
echo "🔍 Available MCP tools:"
echo "- read_query: Execute SELECT queries with validation"
echo "- list_tables: Show all database tables"
echo "- describe_table: Show table schema details"

# Verification
echo ""
echo "🧪 Running verification..."
echo "SQLite Read-Only server details:"
claude mcp get "${MCP_CONFIG_NAME}"
