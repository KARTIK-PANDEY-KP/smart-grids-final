import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import type { FC } from "react";
import {
  ArrowDownIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  PencilIcon,
  RefreshCwIcon,
  SendHorizontalIcon,
  SearchIcon,
  DatabaseIcon,
  TerminalIcon,
  ChevronDownIcon,
  BrainIcon,
  ChevronRightIcon as ChevronRightIconAlias,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState, useEffect, useRef, createContext, useContext, useCallback, useMemo } from "react";

import { Button } from "@/components/ui/button";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";

// Create a context to share tool outputs between components
interface ToolContextType {
  toolOutputs: Array<{
    id: string;
    toolName: string;
    content: string;
    timestamp?: string;
    searchTerm?: string;
  }>;
  addToolOutput: (tool: {toolName: string; content: string; searchTerm?: string}) => void;
  clearToolOutputs: () => void;
}

const ToolContext = createContext<ToolContextType>({
  toolOutputs: [],
  addToolOutput: () => {},
  clearToolOutputs: () => {}
});

// Helper to extract and dispatch tool output events
function useToolOutputExtractor() {
  const { addToolOutput, toolOutputs } = useContext(ToolContext);
  
  useEffect(() => {
    // Function to check for and extract tool outputs from a text node
    const extractToolOutputs = (node: Node) => {
      if (node.nodeType !== Node.TEXT_NODE) return;
      
      const text = node.textContent || '';
      if (!text.includes('<<TOOL_START>>')) return;
      
      // Extract tool outputs using regex
      const regex = /<<TOOL_START>>([\s\S]*?)<<TOOL_END>>/g;
      let match;
      let processedText = text;
      
      while ((match = regex.exec(text)) !== null) {
        if (match[1]) {
          const toolContent = match[1].trim();
          
          // Skip if this exact content already exists
          if (toolOutputs.some(tool => tool.content === toolContent)) {
            // Simply remove the marker without adding a duplicate
            processedText = processedText.replace(match[0], "");
            continue;
          }
          
          // Determine tool name based on content
          let toolName = "Tool Output";
          
          if (toolContent.includes("Searching interview transcripts for")) {
            toolName = "Interview Search";
          } else if (toolContent.includes("Searching for")) {
            toolName = "Web Search";
          } else {
            // Try to extract tool name from the content
            const nameMatch = toolContent.match(/^([a-zA-Z_]+):\s*(.*)/);
            if (nameMatch) {
              toolName = nameMatch[1];
            }
          }
          
          // Extract search term for interview searches (for deduplication)
          let searchTerm: string | undefined = undefined;
          if (toolName === "Interview Search") {
            const searchMatch = toolContent.match(/Searching interview transcripts for ['"](.*?)['"]/i);
            if (searchMatch && searchMatch[1]) {
              searchTerm = searchMatch[1].trim();
              
              // Check if we already have a result for this search term
              const existingSearchOutput = toolOutputs.find(tool => 
                tool.toolName === "Interview Search" && 
                tool.searchTerm === searchTerm
              );
              
              if (existingSearchOutput) {
                // Skip adding a duplicate search result
                processedText = processedText.replace(match[0], "");
                continue;
              }
            }
          }
          
          // Log for debugging
          console.log(`Found tool output: ${toolName}`);
          
          // Add to thinking box
          addToolOutput({ 
            toolName, 
            content: toolContent,
            searchTerm
          });
          
          // Remove this tool message from the displayed text
          processedText = processedText.replace(match[0], "");
        }
      }
      
      // Update the text content if we removed any tool markers
      if (processedText !== text) {
        node.textContent = processedText;
      }
    };
    
    // Create an aggressive checker that runs periodically
    const checkAllText = () => {
      // Find all text nodes in the document
      const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        null
      );
      
      const textNodes = [];
      let node;
      while (node = walker.nextNode()) {
        textNodes.push(node);
      }
      
      // Check each text node for tool output
      textNodes.forEach(extractToolOutputs);
    };
    
    // Create MutationObserver to watch for changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach(mutation => {
        // Check added nodes for tool output
        mutation.addedNodes.forEach(node => {
          // Handle text nodes
          extractToolOutputs(node);
          
          // Handle element nodes and their children
          if (node.nodeType === Node.ELEMENT_NODE) {
            const textNodes = Array.from(node.childNodes).filter(
              child => child.nodeType === Node.TEXT_NODE
            );
            textNodes.forEach(extractToolOutputs);
          }
        });
        
        // Check changed text nodes
        if (mutation.type === 'characterData') {
          extractToolOutputs(mutation.target);
        }
      });
      
      // Also run a full check in case markers were split across nodes
      checkAllText();
    });
    
    // Start observing the entire document for changes
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true
    });
    
    // Also check periodically
    const intervalId = setInterval(checkAllText, 500);
    
    return () => {
      observer.disconnect();
      clearInterval(intervalId);
    };
  }, [addToolOutput, toolOutputs]);
}

// Thinking box component that shows between messages
const ThinkingBox: FC = () => {
  const { toolOutputs } = useContext(ToolContext);
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Skip rendering if no tools
  if (toolOutputs.length === 0) return null;
  
  // Group duplicate tools to avoid showing identical ones
  const uniqueTools = toolOutputs.reduce((acc, tool) => {
    // Check if we already have this exact tool content
    const existingTool = acc.find(t => t.content === tool.content);
    if (existingTool) return acc;
    
    // For interview searches, group by search term
    if (tool.toolName === "Interview Search" && tool.searchTerm) {
      const existingSearchTool = acc.find(t => 
        t.toolName === "Interview Search" && 
        t.searchTerm === tool.searchTerm
      );
      if (existingSearchTool) return acc;
    }
    
    return [...acc, tool];
  }, [] as typeof toolOutputs);
  
  return (
    <div className="w-full max-w-[var(--thread-max-width)] my-2 flex justify-center">
      <div className="bg-slate-50 rounded-lg border border-slate-200 shadow-sm max-w-[80%] w-full overflow-hidden">
        {/* Clickable header to expand/collapse */}
        <div 
          className="flex items-center justify-between px-3 py-2 bg-slate-100 cursor-pointer hover:bg-slate-200 transition-colors"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center gap-2 text-slate-700 font-medium">
            <BrainIcon size={18} />
            <div>Assistant's Thought Process</div>
            <div className="text-xs text-slate-500 font-normal ml-1">
              ({uniqueTools.length} {uniqueTools.length === 1 ? 'tool' : 'tools'} used)
            </div>
          </div>
          
          <div className="text-slate-500">
            {isExpanded ? <ChevronDownIcon size={18} /> : <ChevronRightIcon size={18} />}
          </div>
        </div>
        
        {/* Collapsible content */}
        {isExpanded && (
          <div className="text-sm text-slate-600 space-y-2 p-3">
            {uniqueTools.map((tool) => (
              <div key={tool.id} className="bg-white rounded border border-slate-200 p-2">
                <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700 mb-1">
                  {tool.toolName.toLowerCase().includes('search') ? 
                    <SearchIcon size={12} /> : 
                    tool.toolName.toLowerCase().includes('interview') ? 
                      <DatabaseIcon size={12} /> : 
                      <TerminalIcon size={12} />
                  }
                  <span>{tool.toolName}</span>
                </div>
                <div className="whitespace-pre-wrap text-xs font-mono">
                  {tool.content}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export const Thread: FC = () => {
  // State for tool outputs
  const [toolOutputs, setToolOutputs] = useState<Array<{
    id: string;
    toolName: string;
    content: string;
    timestamp?: string;
    searchTerm?: string;
  }>>([]);
  
  // Track if the assistant is responding
  const [isAssistantResponding, setIsAssistantResponding] = useState(false);
  
  // Reset state when the component mounts (to clear any lingering state)
  useEffect(() => {
    setToolOutputs([]);
    setIsAssistantResponding(false);
  }, []);
  
  // Add tool output function - memoized with useCallback
  const addToolOutput = useCallback((tool: {toolName: string; content: string; searchTerm?: string}) => {
    const id = `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Don't add duplicate tools 
    setToolOutputs(prev => {
      // Check if this exact content already exists
      if (prev.some(t => t.content === tool.content)) {
        return prev;
      }
      
      // For interview searches, check if we already have this search term
      if (tool.toolName === "Interview Search" && tool.searchTerm) {
        if (prev.some(t => 
          t.toolName === "Interview Search" && 
          t.searchTerm === tool.searchTerm
        )) {
          return prev;
        }
      }
      
      return [...prev, { 
        id,
        toolName: tool.toolName, 
        content: tool.content, 
        timestamp: new Date().toLocaleTimeString(),
        searchTerm: tool.searchTerm
      }];
    });
    
    // When tools are being used, the assistant is responding
    setIsAssistantResponding(true);
  }, []);
  
  // Clear tool outputs function - memoized with useCallback
  const clearToolOutputs = useCallback(() => {
    setToolOutputs([]);
    setIsAssistantResponding(false);
  }, []);
  
  // Memoized context value to prevent unnecessary re-renders
  const contextValue = useMemo(() => ({
    toolOutputs,
    addToolOutput,
    clearToolOutputs
  }), [toolOutputs, addToolOutput, clearToolOutputs]);
  
  // Add ref for scrolling
  const viewportRef = useRef<HTMLDivElement>(null);
  
  // Scroll to bottom whenever messages change
  useEffect(() => {
    // Short delay to ensure content is rendered
    const timer = setTimeout(() => {
      if (viewportRef.current) {
        viewportRef.current.scrollTop = viewportRef.current.scrollHeight;
      }
    }, 100);
    
    return () => clearTimeout(timer);
  }, [toolOutputs, isAssistantResponding]);
  
  return (
    <ToolContext.Provider value={contextValue}>
      <ThreadPrimitive.Root
        className="bg-background box-border flex h-full flex-col overflow-hidden"
        style={{
          ["--thread-max-width" as string]: "42rem",
        }}
      >
        <ThreadPrimitive.Viewport 
          ref={viewportRef}
          className="flex h-full flex-col items-center overflow-y-auto scroll-smooth bg-inherit px-4 pt-8"
        >
          <ThreadWelcome />

          <ThreadPrimitive.Messages
            components={{
              UserMessage: UserMessage,
              EditComposer: EditComposer,
              AssistantMessage: AssistantMessage,
            }}
          />

          {/* Only show thinking box when assistant is responding and there are tool outputs */}
          {isAssistantResponding && toolOutputs.length > 0 && <ThinkingBox />}

          <ThreadPrimitive.If empty={false}>
            <div className="min-h-8 flex-grow" />
          </ThreadPrimitive.If>

          <div className="sticky bottom-0 mt-3 flex w-full max-w-[var(--thread-max-width)] flex-col items-center justify-end rounded-t-lg bg-inherit pb-4">
            <ThreadScrollToBottom />
            <Composer />
          </div>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </ToolContext.Provider>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="absolute -top-8 rounded-full disabled:invisible"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  return (
    <ThreadPrimitive.Empty>
      <div className="flex w-full max-w-[var(--thread-max-width)] flex-grow flex-col">
        <div className="flex w-full flex-grow flex-col items-center justify-center">
          <p className="mt-4 font-medium">How can I help you today?</p>
        </div>
        <ThreadWelcomeSuggestions />
      </div>
    </ThreadPrimitive.Empty>
  );
};

const ThreadWelcomeSuggestions: FC = () => {
  return (
    <div className="mt-3 flex w-full items-stretch justify-center gap-4">
      <ThreadPrimitive.Suggestion
        className="hover:bg-muted/80 flex max-w-sm grow basis-0 flex-col items-center justify-center rounded-lg border p-3 transition-colors ease-in"
        prompt="What is the weather in Tokyo?"
        method="replace"
        autoSend
      >
        <span className="line-clamp-2 text-ellipsis text-sm font-semibold">
          What is the weather in Tokyo?
        </span>
      </ThreadPrimitive.Suggestion>
      <ThreadPrimitive.Suggestion
        className="hover:bg-muted/80 flex max-w-sm grow basis-0 flex-col items-center justify-center rounded-lg border p-3 transition-colors ease-in"
        prompt="What is assistant-ui?"
        method="replace"
        autoSend
      >
        <span className="line-clamp-2 text-ellipsis text-sm font-semibold">
          What is assistant-ui?
        </span>
      </ThreadPrimitive.Suggestion>
    </div>
  );
};

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="focus-within:border-ring/20 flex w-full flex-wrap items-end rounded-lg border bg-inherit px-2.5 shadow-sm transition-colors ease-in">
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Write a message..."
        className="placeholder:text-muted-foreground max-h-40 flex-grow resize-none border-none bg-transparent px-2 py-4 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
      />
      <ComposerAction />
    </ComposerPrimitive.Root>
  );
};

const ComposerAction: FC = () => {
  return (
    <>
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send asChild>
          <TooltipIconButton
            tooltip="Send"
            variant="default"
            className="my-2.5 size-8 p-2 transition-opacity ease-in"
          >
            <SendHorizontalIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel asChild>
          <TooltipIconButton
            tooltip="Cancel"
            variant="default"
            className="my-2.5 size-8 p-2 transition-opacity ease-in"
          >
            <CircleStopIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
    </>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="grid auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] gap-y-2 [&:where(>*)]:col-start-2 w-full max-w-[var(--thread-max-width)] py-4">
      <UserActionBar />

      <div className="bg-muted text-foreground max-w-[calc(var(--thread-max-width)*0.8)] break-words rounded-3xl px-5 py-2.5 col-start-2 row-start-2">
        <MessagePrimitive.Content />
      </div>

      <BranchPicker className="col-span-full col-start-1 row-start-3 -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      className="flex flex-col items-end col-start-1 row-start-2 mr-3 mt-2.5"
    >
      <ActionBarPrimitive.Edit asChild>
        <TooltipIconButton tooltip="Edit">
          <PencilIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  return (
    <ComposerPrimitive.Root className="bg-muted my-4 flex w-full max-w-[var(--thread-max-width)] flex-col gap-2 rounded-xl">
      <ComposerPrimitive.Input className="text-foreground flex h-8 w-full resize-none bg-transparent p-4 pb-0 outline-none" />

      <div className="mx-3 mb-3 flex items-center justify-center gap-2 self-end">
        <ComposerPrimitive.Cancel asChild>
          <Button variant="ghost">Cancel</Button>
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send asChild>
          <Button>Send</Button>
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  // Use the tool output extractor hook
  useToolOutputExtractor();
  
  // Get the clearToolOutputs function from context
  const { clearToolOutputs } = useContext(ToolContext);
  
  // Use a ref to track whether this is the first render
  const isFirstRender = useRef(true);
  
  // Only clear tool outputs on first render, not on every render
  useEffect(() => {
    if (isFirstRender.current) {
      clearToolOutputs();
      isFirstRender.current = false;
    }
  }, [clearToolOutputs]);
  
  return (
    <MessagePrimitive.Root className="grid grid-cols-[auto_auto_1fr] grid-rows-[auto_1fr] relative w-full max-w-[var(--thread-max-width)] py-4">
      <div className="text-foreground max-w-[calc(var(--thread-max-width)*0.8)] break-words leading-7 col-span-2 col-start-2 row-start-1 my-1.5">
        <MessagePrimitive.Content components={{ Text: MarkdownText }} />
      </div>

      <AssistantActionBar />

      <BranchPicker className="col-start-2 row-start-2 -ml-2 mr-2" />
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="text-muted-foreground flex gap-1 col-start-3 row-start-2 -ml-1 data-[floating]:bg-background data-[floating]:absolute data-[floating]:rounded-md data-[floating]:border data-[floating]:p-1 data-[floating]:shadow-sm"
    >
      <ActionBarPrimitive.Copy asChild>
        <TooltipIconButton tooltip="Copy">
          <MessagePrimitive.If copied>
            <CheckIcon />
          </MessagePrimitive.If>
          <MessagePrimitive.If copied={false}>
            <CopyIcon />
          </MessagePrimitive.If>
        </TooltipIconButton>
      </ActionBarPrimitive.Copy>
      <ActionBarPrimitive.Reload asChild>
        <TooltipIconButton tooltip="Refresh">
          <RefreshCwIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Reload>
    </ActionBarPrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn(
        "text-muted-foreground inline-flex items-center text-xs",
        className
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton tooltip="Previous">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip="Next">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};

const CircleStopIcon = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      width="16"
      height="16"
    >
      <rect width="10" height="10" x="3" y="3" rx="2" />
    </svg>
  );
};
