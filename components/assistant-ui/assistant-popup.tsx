"use client";

import { useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { 
  Dialog, 
  DialogContent, 
  DialogTrigger,
  DialogTitle
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { MessageCircle } from "lucide-react";

export const AssistantPopup = () => {
  const [open, setOpen] = useState(false);
  const runtime = useChatRuntime({
    api: "http://localhost:8000/api/my-custom-chat",
  });

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full p-0 shadow-lg"
          size="icon"
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px] max-h-[80vh] h-[600px] p-0 flex flex-col">
        <DialogTitle className="sr-only">AI Assistant Chat</DialogTitle>
        <AssistantRuntimeProvider runtime={runtime}>
          <div className="flex flex-col h-full overflow-hidden">
            <div className="flex-none p-3 border-b">
              <h3 className="text-lg font-medium">AI Assistant</h3>
            </div>
            <div className="flex-grow relative overflow-hidden">
              <Thread />
            </div>
          </div>
        </AssistantRuntimeProvider>
      </DialogContent>
    </Dialog>
  );
}; 