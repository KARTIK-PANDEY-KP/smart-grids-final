# SmartGrid AI

SmartGrid AI is a spreadsheet-like web application that allows users to enrich structured data (like student info, leads, etc.) with AI-powered columns using LLMs (OpenAI GPT). No coding required—just type, click, and get insights!

---

## Features
- Spreadsheet UI: Add, edit, delete rows and columns
- AI Columns: Enrich data with AI (classification, extraction, etc.)
- Regenerate AI results per row
- Save/load sheets (localStorage)
- Export to CSV
- Multiple sheet templates (e.g., Student Info, Leads)
- FastAPI backend with OpenAI integration

---

## Tech Stack
- **Frontend:** React (Next.js), TailwindCSS, TypeScript
- **Backend:** FastAPI (Python), OpenAI API

---

## Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd smart-grid-ai
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/` with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Start the backend:
```bash
uvicorn app.main:app --reload
```

### 3. Frontend Setup
```bash
cd ../  # Project root
npm install  # or pnpm install
npm run dev  # or pnpm dev
```

Visit [http://localhost:3000/app](http://localhost:3000/app) in your browser.

---

## Usage
- Click **Add AI Column** to create a new AI-powered column.
- Enter a prompt (e.g., "Classify this major as Engineer or Non-Engineer").
- Click **Regenerate** to update AI results for a row after editing.
- Click **Export** to download the current sheet as CSV.

---

## API Endpoints
- `POST /api/enrich` — Enrich a single row with AI
- `POST /api/enrich/batch` — Enrich multiple rows at once

---

## License
MIT 

# Smart Sheet Chat Database Documentation

## Overview

This document outlines the database implementation for the chat feature in Smart Sheet. The chat system allows users to interact with an AI assistant while preserving conversation history.

## Database Schema

We use SQLite with the following tables:

### Chats Table

```sql
CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    title TEXT,
    model TEXT NOT NULL,
    chat_metadata TEXT
);
```

| Column | Type | Description | Rationale |
|--------|------|-------------|-----------|
| `id` | TEXT | Unique identifier for the chat (UUID) | UUIDs allow for distributed generation without collisions |
| `created_at` | TIMESTAMP | When the chat was created | Enables time-based sorting and analytics |
| `title` | TEXT | User-friendly name for the chat | Helps users identify conversations |
| `model` | TEXT | AI model used (e.g., "gpt-4o") | Records which model was used for reproducibility |
| `chat_metadata` | TEXT | JSON string for additional settings | Flexible storage for system prompts and settings |

### Chat Messages Table

```sql
CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT REFERENCES chats(id) ON DELETE CASCADE,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_name TEXT,
    tool_args TEXT,
    partial_json TEXT
);
```

| Column | Type | Description | Rationale |
|--------|------|-------------|-----------|
| `id` | INTEGER | Auto-incrementing message ID | Simple, numeric identifiers for messages |
| `chat_id` | TEXT | Foreign key to chats table | Links messages to their parent chat |
| `ts` | TIMESTAMP | When the message was created | Maintains chronological order of messages |
| `role` | TEXT | Who sent the message (user/assistant/tool) | Distinguishes between different participants |
| `content` | TEXT | The actual message content | Stores the text of each message |
| `tool_name` | TEXT | Name of tool used, if applicable | Records which tools were called (e.g., web_search) |
| `tool_args` | TEXT | JSON string of tool arguments | Preserves tool call parameters for reproducibility |
| `partial_json` | TEXT | For storing streaming/partial responses | Enables support for streaming responses |

## Design Decisions

### Why SQLite?

1. **Simplicity**: SQLite requires no separate server, making it easy to set up and maintain.
2. **Portability**: The entire database is contained in a single file, easy to backup or move.
3. **Performance**: For chat applications with moderate load, SQLite provides sufficient performance.
4. **Zero Configuration**: Works out of the box without complex configuration.

### Entity Relationships

- **One-to-Many Relationship**: Each chat can have many messages, implemented with a foreign key from `chat_messages.chat_id` to `chats.id`.
- **Cascading Deletion**: When a chat is deleted, all its messages are automatically deleted via `ON DELETE CASCADE`.

### Content Storage

- **JSON in Text Fields**: We store structured data (metadata, tool arguments) as JSON strings in TEXT fields for flexibility.
- **Denormalized Tool Data**: Tool calls are stored in the same table as messages for simplicity and ease of retrieval.

### ID Strategy

- **UUIDs for Chats**: We use UUIDs for chat IDs to allow for distributed generation without coordination.
- **Auto-increment for Messages**: Simple numeric IDs are sufficient for messages since they only need to be unique within their parent chat.

## Query Examples

### Get Last 5 Messages (Newest First)

```sql
SELECT substr(chat_id, -8) as chat_id, role, 
       substr(content, 1, 60) as message_preview, 
       datetime(ts) as timestamp 
FROM chat_messages 
ORDER BY ts DESC 
LIMIT 5;
```

### List All Chats

```sql
SELECT id, substr(id, -8) as short_id, title, created_at 
FROM chats 
ORDER BY created_at DESC;
```

### Get Messages for a Specific Chat

```sql
SELECT role, content, datetime(ts) as timestamp 
FROM chat_messages 
WHERE chat_id = ? 
ORDER BY ts ASC;
```

## Implementation Considerations

1. **Chat Session Persistence**: Each chat session gets a unique ID that's passed between frontend and backend.
2. **Metadata Storage**: System prompts and other settings are stored as JSON in the `chat_metadata` field.
3. **Tool Integration**: All tool calls and their results are tracked in the database for context preservation.
4. **Message Chronology**: Timestamps ensure proper ordering of messages for playback and context building.

## Extensibility

The schema was designed to be flexible and allow for future enhancements:

1. **Additional Message Types**: The `role` field can accommodate new participant types.
2. **Extended Metadata**: The JSON-based metadata fields can store additional properties without schema changes.
3. **Multiple Models**: Different AI models can be used and tracked via the `model` field.

## Limitations

1. **No Native JSON Support**: SQLite lacks native JSON support, so complex queries on JSON fields require parsing.
2. **No Real-time Events**: There's no built-in notification system; polling or external solutions are needed.
3. **Single-file Concurrency**: Heavy write loads may cause contention due to SQLite's file-based locking.

## Usage in the Application

The chat database is used for:

1. **Message History**: Preserving all user and assistant messages for continuity.
2. **Context Management**: Providing the AI with sufficient context to maintain coherent conversations.
3. **Tool Call Tracking**: Recording tool usage and results for debugging and transparency.
4. **Analytics**: Enabling analysis of conversation patterns and assistant performance. 