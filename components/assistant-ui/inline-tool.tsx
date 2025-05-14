import { FC, useState } from "react";
import { SearchIcon, DatabaseIcon, TerminalIcon, ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import { cn } from "@/lib/utils";

// Inline tool output component
export const InlineTool: FC<{
  toolName: string;
  content: string;
}> = ({ toolName, content }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Get icon for the tool type
  const getToolIcon = () => {
    const name = toolName.toLowerCase();
    
    if (name.includes('search') || name.includes('web')) {
      return <SearchIcon size={14} className="mr-1.5" />;
    } else if (name.includes('interview') || name.includes('database')) {
      return <DatabaseIcon size={14} className="mr-1.5" />;
    } else {
      return <TerminalIcon size={14} className="mr-1.5" />;
    }
  };
  
  // Format the content based on tool type
  const formatContent = () => {
    // For interview search results
    if (toolName.toLowerCase().includes('interview')) {
      let formatted = content;
      
      // Format section numbers [1], [2], etc.
      formatted = formatted.replace(/\[(\d+)\]\s+\[([^\]]+)\]/g, 
        '<div class="font-semibold mt-2 text-blue-600">[Result $1] <span class="text-blue-800">[$2]</span></div>');
      
      // Format speaker tags
      formatted = formatted.replace(/SPEAKER_(\d+):/g, 
        '<span class="font-semibold text-emerald-700">Speaker $1:</span>');
      
      // Add spacing for readability
      formatted = formatted.replace(/\n/g, '<br/>');
      
      return <div dangerouslySetInnerHTML={{ __html: formatted }} />;
    }
    
    // For web search results
    if (toolName.toLowerCase().includes('search')) {
      let formatted = content;
      
      // Add paragraph spacing
      formatted = formatted.replace(/\n\n/g, '<br/><br/>');
      formatted = formatted.replace(/\n/g, '<br/>');
      
      // Highlight search terms (if any)
      const searchTerms = content.match(/Searching for (.+?)\.\.\./);
      if (searchTerms && searchTerms[1]) {
        const terms = searchTerms[1].split(' ').filter(t => t.length > 3);
        terms.forEach(term => {
          formatted = formatted.replace(new RegExp(`\\b${term}\\b`, 'gi'), 
            `<span class="bg-yellow-100 text-yellow-800 px-1 rounded-sm">$&</span>`);
        });
      }
      
      return <div dangerouslySetInnerHTML={{ __html: formatted }} />;
    }
    
    // For all other tool outputs, add line breaks but keep as plain text
    return content.split('\n').map((line, index) => (
      <div key={index}>{line || <br/>}</div>
    ));
  };
  
  return (
    <div className="mb-3 w-full overflow-hidden rounded-md border border-slate-200 shadow-sm">
      <div 
        className={cn(
          "flex cursor-pointer items-center px-3 py-2 text-sm transition-colors",
          isExpanded 
            ? "bg-slate-200 text-slate-800 font-medium" 
            : "bg-slate-100 text-slate-700 hover:bg-slate-150"
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="mr-1.5">
          {isExpanded ? <ChevronDownIcon size={16} /> : <ChevronRightIcon size={16} />}
        </div>
        <div className="flex items-center flex-grow">
          {getToolIcon()}
          {toolName}
        </div>
      </div>
      
      {isExpanded && (
        <div className="border-t border-slate-200 bg-white p-3 text-sm text-slate-700 max-h-[300px] overflow-auto">
          {formatContent()}
        </div>
      )}
    </div>
  );
}; 