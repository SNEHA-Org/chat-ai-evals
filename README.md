# DSPy Optimization Dashboard

A real-time web interface for visualizing and comparing DSPy prompt optimization results.

## 🌟 Features

- **Real-time optimization progress** - WebSocket-based live updates during DSPy optimization
- **Golden dataset viewer** - Interactive table to browse training examples
- **Prompt version management** - Automatic versioning with timestamps in `/data/prompts`
- **Side-by-side prompt comparison** - Visual diff between original and optimized prompts
- **Optimization metrics** - Before/after scores, improvement percentage, iteration counts
- **Activity logging** - Real-time activity feed with timestamps

## 🏗️ Architecture

```
┌─── Frontend (React + TypeScript) ───┐    ┌─── Backend (FastAPI + Socket.IO) ───┐
│                                     │    │                                     │
│  • Dashboard Component              │◄──►│  • WebSocket Server                 │
│  • Real-time Updates via Socket.IO │    │  • REST API Endpoints              │
│  • Golden Dataset Table            │    │  • DSPy Integration                 │
│  • Prompt Comparison Modal         │    │  • File Management                  │
│  • Progress Visualization          │    │                                     │
│                                     │    │                                     │
└─────────────────────────────────────┘    └─────────────────────────────────────┘
                     │                                        │
                     │                                        │
                     ▼                                        ▼
            ┌─── Data Directory ───┐                ┌─── DSPy Optimizer ───┐
            │                     │                │                       │
            │  /data/examples/    │                │  • COPRO Algorithm    │
            │  /data/prompts/     │                │  • Progress Callbacks │
            │                     │                │  • Model Evaluation   │
            └─────────────────────┘                └───────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- OpenAI API key
- Vector store ID for file search

### Installation

1. **Clone and setup Python environment:**
   ```bash
   cd chat-ai-evals
   pip install -r requirements.txt
   ```

2. **Setup frontend:**
   ```bash
   cd ui
   npm install
   ```

3. **Configure environment:**
   ```bash
   # Create .env file in project root
   OPENAI_API_KEY=your_api_key_here
   VECTOR_STORE_ID=vs_xxxxx
   ```

4. **Prepare data:**
   ```bash
   # Ensure your golden dataset is in the right location
   # File should be: data/examples/examples.csv
   # Format: no,question,reference_answer
   ```

### Running the System

**Option 1: Use the startup script (Recommended)**
```bash
python start_system.py
```

**Option 2: Manual startup**
```bash
# Terminal 1 - Backend
python server.py

# Terminal 2 - Frontend  
cd ui && npm run dev
```

**Access the dashboard:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- WebSocket: ws://localhost:8000/socket.io/

## 📁 Data Directory Structure

```
data/
├── examples/
│   └── examples.csv          # Golden dataset (no, question, reference_answer)
└── prompts/
    ├── prompt_current.md     # Current/original prompt
    └── optimized_YYYYMMDD_HHMMSS.md  # Auto-generated optimized prompts
```

## 🎛️ Dashboard Components

### 1. Optimization Progress Panel
- **Status indicators**: Idle, Running, Completed, Error
- **Progress bar**: Real-time percentage completion
- **Current step**: Live updates of optimization phase
- **Duration tracking**: Time elapsed since start
- **Start button**: Trigger new optimization run

### 2. Golden Dataset Viewer
- **Interactive table**: Sortable, searchable, paginated
- **Example count**: Total number of training examples
- **Content preview**: Truncated display with full content on hover
- **CSV metadata**: File information and statistics

### 3. Prompt Versions Panel
- **Version listing**: All prompts with timestamps and types
- **File metadata**: Size, modification date, type badges
- **Latest indicator**: Highlights most recent version
- **Quick actions**: Refresh list, trigger comparison

### 4. Side-by-side Comparison Modal
- **Prompt selection**: Dropdown menus for version picking
- **Synchronized scrolling**: Compare content line-by-line
- **Metadata display**: File stats, modification dates
- **Difference metrics**: Line and character count differences

### 5. Results Dashboard
- **Before/after scores**: Original vs optimized performance
- **Improvement percentage**: Calculated performance gain
- **Iteration count**: Number of optimization steps
- **Example usage**: Total training examples processed

### 6. Activity Log
- **Real-time updates**: Live WebSocket-powered activity feed
- **Timestamps**: Precise timing for all events
- **Error tracking**: Failed operations with details
- **Auto-scroll**: Always shows latest activity

## 🔌 WebSocket Events

### Client → Server
- `start_optimization`: Trigger new optimization run
- `get_golden_dataset`: Request dataset refresh
- `get_prompt_versions`: Request prompt list refresh  
- `compare_prompts`: Request prompt comparison data

### Server → Client
- `state_update`: Current optimization status
- `optimization_started`: Optimization beginning
- `progress_update`: Real-time progress updates
- `optimization_completed`: Successful completion
- `optimization_error`: Error during optimization
- `golden_dataset`: Dataset content and metadata
- `prompt_versions`: Available prompt files
- `prompt_comparison`: Side-by-side comparison data

## 🛠️ Development

### Frontend Stack
- **React 18** with TypeScript
- **Vite** for fast development builds
- **Tailwind CSS** for utility-first styling
- **TanStack Table** for data grid functionality
- **Socket.IO Client** for real-time communication
- **Lucide React** for icons

### Backend Stack
- **FastAPI** for REST API endpoints
- **Python Socket.IO** for WebSocket handling
- **DSPy** for prompt optimization
- **Pandas** for data manipulation
- **OpenAI** for LLM integration

### Key Files
```
├── server.py                     # FastAPI + Socket.IO server
├── evals/rag_with_events.py     # DSPy optimization with callbacks
├── ui/src/
│   ├── hooks/useSocket.ts       # WebSocket connection management
│   ├── components/Dashboard.tsx # Main dashboard component
│   └── components/              # Individual UI components
├── data/                        # Data directory
└── start_system.py             # Startup script
```

## 🐛 Troubleshooting

### Common Issues

**1. Backend won't start**
```bash
# Check dependencies
pip install -r requirements.txt

# Verify environment variables
cat .env

# Check port availability
lsof -i :8000
```

**2. Frontend build errors**
```bash
# Clear cache and reinstall
cd ui
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 18+
```

**3. WebSocket connection failed**
- Ensure backend is running on port 8000
- Check CORS settings in server.py
- Verify firewall settings

**4. Optimization fails**
- Verify OpenAI API key is valid
- Check vector store ID exists
- Ensure golden dataset is properly formatted

**5. Data not loading**
- Verify `data/examples/examples.csv` exists
- Check CSV format: `no,question,reference_answer`
- Ensure proper file permissions