import type React from "react";
import type { Message } from "@langchain/langgraph-sdk";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Copy, CopyCheck } from "lucide-react";
import { InputForm } from "@/components/InputForm";
import { Button } from "@/components/ui/button";
import { useState, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  ActivityTimeline,
  ProcessedEvent,
} from "@/components/ActivityTimeline"; // Assuming ActivityTimeline is in the same dir or adjust path

// Markdown component props type from former ReportView
type MdComponentProps = {
  className?: string;
  children?: ReactNode;
  [key: string]: any;
};

// Add types for RAG attribution data
interface RagAttributionData {
  attributions?: Array<{
    content_ids?: string[];
  }>;
  retrieval_contents?: Array<{
    content_id: string;
    number: string | number;
  }>;
  message_id?: string;
  agent_id?: string;
}

// Markdown components (from former ReportView.tsx)
const mdComponents = {
  h1: ({ className, children, ...props }: MdComponentProps) => (
    <h1 className={cn("text-2xl font-bold mt-4 mb-2", className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: MdComponentProps) => (
    <h2 className={cn("text-xl font-bold mt-3 mb-2", className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: MdComponentProps) => (
    <h3 className={cn("text-lg font-bold mt-3 mb-1", className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: MdComponentProps) => (
    <p className={cn("mb-3 leading-7", className)} {...props}>
      {children}
    </p>
  ),
  a: ({ className, children, href, ...props }: MdComponentProps) => (
    <Badge className="text-xs mx-0.5">
      <a
        className={cn("text-blue-400 hover:text-blue-300 text-xs", className)}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        {...props}
      >
        {children}
      </a>
    </Badge>
  ),
  ul: ({ className, children, ...props }: MdComponentProps) => (
    <ul className={cn("list-disc pl-6 mb-3", className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: MdComponentProps) => (
    <ol className={cn("list-decimal pl-6 mb-3", className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: MdComponentProps) => (
    <li className={cn("mb-1", className)} {...props}>
      {children}
    </li>
  ),
  blockquote: ({ className, children, ...props }: MdComponentProps) => (
    <blockquote
      className={cn(
        "border-l-4 border-neutral-600 pl-4 italic my-3 text-sm",
        className
      )}
      {...props}
    >
      {children}
    </blockquote>
  ),
  code: ({ className, children, ...props }: MdComponentProps) => (
    <code
      className={cn(
        "bg-neutral-900 rounded px-1 py-0.5 font-mono text-xs",
        className
      )}
      {...props}
    >
      {children}
    </code>
  ),
  pre: ({ className, children, ...props }: MdComponentProps) => (
    <pre
      className={cn(
        "bg-neutral-900 p-3 rounded-lg overflow-x-auto font-mono text-xs my-3",
        className
      )}
      {...props}
    >
      {children}
    </pre>
  ),
  hr: ({ className, ...props }: MdComponentProps) => (
    <hr className={cn("border-neutral-600 my-4", className)} {...props} />
  ),
  table: ({ className, children, ...props }: MdComponentProps) => (
    <div className="my-3 overflow-x-auto">
      <table className={cn("border-collapse w-full", className)} {...props}>
        {children}
      </table>
    </div>
  ),
  th: ({ className, children, ...props }: MdComponentProps) => (
    <th
      className={cn(
        "border border-neutral-600 px-3 py-2 text-left font-bold",
        className
      )}
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ className, children, ...props }: MdComponentProps) => (
    <td
      className={cn("border border-neutral-600 px-3 py-2", className)}
      {...props}
    >
      {children}
    </td>
  ),
};

// Props for HumanMessageBubble
interface HumanMessageBubbleProps {
  message: Message;
  mdComponents: typeof mdComponents;
}

// HumanMessageBubble Component
const HumanMessageBubble: React.FC<HumanMessageBubbleProps> = ({
  message,
  mdComponents,
}) => {
  return (
    <div
      className={`text-white rounded-3xl break-words min-h-7 bg-neutral-700 max-w-[100%] sm:max-w-[90%] px-4 pt-3 rounded-br-lg`}
    >
      <ReactMarkdown components={mdComponents}>
        {typeof message.content === "string"
          ? message.content
          : JSON.stringify(message.content)}
      </ReactMarkdown>
    </div>
  );
};

// Props for AiMessageBubble
interface AiMessageBubbleProps {
  message: Message;
  historicalActivity: ProcessedEvent[] | undefined;
  liveActivity: ProcessedEvent[] | undefined;
  isLastMessage: boolean;
  isOverallLoading: boolean;
  mdComponents: typeof mdComponents;
  handleCopy: (text: string, messageId: string) => void;
  copiedMessageId: string | null;
  handleAttributionClick: (contentId: string, agentId: string, messageId: string) => void;
}

// AiMessageBubble Component
const AiMessageBubble: React.FC<AiMessageBubbleProps> = ({
  message,
  historicalActivity,
  liveActivity,
  isLastMessage,
  isOverallLoading,
  mdComponents,
  handleCopy,
  copiedMessageId,
  handleAttributionClick,
}) => {
  // Determine which activity events to show and if it's for a live loading message
  const activityForThisBubble =
    isLastMessage && isOverallLoading ? liveActivity : historicalActivity;
  const isLiveActivityForThisBubble = isLastMessage && isOverallLoading;

  // Debug: Log the message object to see what data is available
  console.log("Debug: Message object:", message);
  console.log("Debug: Message keys:", Object.keys(message));

  // Extract RAG attribution data from message if available
  const ragAttributions = (message as any)?.additional_kwargs?.rag_attributions || [];
  const ragRetrievalContents = (message as any)?.additional_kwargs?.rag_retrieval_contents || [];
  const ragMessageId = (message as any)?.additional_kwargs?.rag_message_id;
  const ragAgentId = (message as any)?.additional_kwargs?.rag_agent_id;

  // Debug: Log extracted RAG data
  console.log("Debug: ragAttributions:", ragAttributions);
  console.log("Debug: ragRetrievalContents:", ragRetrievalContents);
  console.log("Debug: ragMessageId:", ragMessageId);
  console.log("Debug: ragAgentId:", ragAgentId);

  // Process unique attributions - ragAttributions is an array of content_id strings
  const uniqueAttributions: { content_id: string; number: string | number }[] = [];
  const seen = new Set<string>();
  
  if (ragAttributions && ragAttributions.length > 0) {
    // ragAttributions is already an array of content_id strings
    ragAttributions.forEach((contentId: string) => {
      if (!seen.has(contentId)) {
        seen.add(contentId);
        const retrieval = ragRetrievalContents.find((rc: any) => rc.content_id === contentId);
        uniqueAttributions.push({ 
          content_id: contentId, 
          number: retrieval ? retrieval.number : contentId 
        });
      }
    });
  }

  console.log("Debug: uniqueAttributions:", uniqueAttributions);

  const messageContent = typeof message.content === "string" 
    ? message.content 
    : JSON.stringify(message.content);

  return (
    <div className={`relative break-words flex flex-col`}>
      {activityForThisBubble && activityForThisBubble.length > 0 && (
        <div className="mb-3 border-b border-neutral-700 pb-3 text-xs">
          <ActivityTimeline
            processedEvents={activityForThisBubble}
            isLoading={isLiveActivityForThisBubble}
          />
        </div>
      )}
      
      <ReactMarkdown 
        components={{
          ...mdComponents,
          // Override the link component to handle RAG citations differently
          a: ({ children, href, ...props }) => {
            // Check if this is a RAG citation (might have different format)
            if (href && href.includes('contextual') || !href) {
              return <span className="text-orange-500 bg-yellow-50 rounded px-1 font-bold mx-1">{children}</span>;
            }
            // Default web citation handling
            return (
              <Badge className="text-xs mx-0.5">
                <a
                  className="text-blue-400 hover:text-blue-300 text-xs"
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  {...props}
                >
                  {children}
                </a>
              </Badge>
            );
          },
          sup: ({ children }) => (
            <sup className="text-orange-500 bg-yellow-50 rounded px-1 font-bold mx-1">
              {children}
            </sup>
          ),
        }}
      >
        {messageContent}
      </ReactMarkdown>

      {/* RAG Attribution buttons */}
      {uniqueAttributions.length > 0 && ragAgentId && (
        <div className="flex flex-row gap-2 mt-3 mb-2 select-none">
          <span className="text-xs text-neutral-400 mr-2">Sources:</span>
          {uniqueAttributions.map((attr) => {
            // Find the retrieval content for this attribution to get the correct message_id
            const retrievalContent = ragRetrievalContents.find((rc: any) => rc.content_id === attr.content_id);
            const messageIdForThisAttribution = retrievalContent?.message_id || ragMessageId;
            
            return (
              <button
                key={attr.content_id}
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  handleAttributionClick(attr.content_id, ragAgentId, messageIdForThisAttribution);
                }}
                className="text-orange-500 bg-orange-50 hover:bg-orange-100 rounded px-2 py-1 text-xs font-medium transition-colors cursor-pointer border border-orange-200"
                title={`Show content for ID: ${attr.content_id}`}
              >
                {attr.number}
              </button>
            );
          })}
        </div>
      )}

      <Button
        variant="default"
        className="cursor-pointer bg-neutral-700 border-neutral-600 text-neutral-300 self-end mt-2"
        onClick={() => handleCopy(messageContent, message.id!)}
      >
        {copiedMessageId === message.id ? "Copied" : "Copy"}
        {copiedMessageId === message.id ? <CopyCheck /> : <Copy />}
      </Button>
    </div>
  );
};

interface ChatMessagesViewProps {
  messages: Message[];
  isLoading: boolean;
  scrollAreaRef: React.RefObject<HTMLDivElement | null>;
  onSubmit: (inputValue: string, effort: string, model: string) => void;
  onCancel: () => void;
  liveActivityEvents: ProcessedEvent[];
  historicalActivities: Record<string, ProcessedEvent[]>;
}

export function ChatMessagesView({
  messages,
  isLoading,
  scrollAreaRef,
  onSubmit,
  onCancel,
  liveActivityEvents,
  historicalActivities,
}: ChatMessagesViewProps) {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [screenshot, setScreenshot] = useState<string | null>(null);

  const handleCopy = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  const handleAttributionClick = async (contentId: string, agentId: string, messageId: string) => {
    console.log("handleAttributionClick: agentId=", agentId, "msgId=", messageId, "contentId=", contentId);
    if (!messageId || !agentId) {
      alert("Missing message_id or agent_id in API response.");
      return;
    }
    try {
      const apiUrl = import.meta.env.DEV 
        ? "http://localhost:2024" 
        : "http://localhost:8123";
      const url = `${apiUrl}/api/retrieval-info?agent_id=${encodeURIComponent(agentId)}&message_id=${encodeURIComponent(messageId)}&content_id=${encodeURIComponent(contentId)}`;
      const res = await fetch(url);
      const data = await res.json();
      console.log("Retrieval info response:", data);
      
      // Updated to handle new response format
      const screenshotBase64 = data.page_image;
      if (screenshotBase64) {
        setScreenshot(screenshotBase64);
      } else {
        setScreenshot(null);
        alert("No screenshot or page image found in content metadata.");
      }
    } catch (err) {
      setScreenshot(null);
      alert("Failed to fetch content metadata.");
    }
  };

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-grow" ref={scrollAreaRef}>
        <div className="p-4 md:p-6 space-y-2 max-w-4xl mx-auto pt-16">
          {messages.map((message, index) => {
            const isLast = index === messages.length - 1;
            return (
              <div key={message.id || `msg-${index}`} className="space-y-3">
                <div
                  className={`flex items-start gap-3 ${
                    message.type === "human" ? "justify-end" : ""
                  }`}
                >
                  {message.type === "human" ? (
                    <HumanMessageBubble
                      message={message}
                      mdComponents={mdComponents}
                    />
                  ) : (
                    <AiMessageBubble
                      message={message}
                      historicalActivity={historicalActivities[message.id!]}
                      liveActivity={liveActivityEvents} // Pass global live events
                      isLastMessage={isLast}
                      isOverallLoading={isLoading} // Pass global loading state
                      mdComponents={mdComponents}
                      handleCopy={handleCopy}
                      copiedMessageId={copiedMessageId}
                      handleAttributionClick={handleAttributionClick}  // Add attribution handler
                    />
                  )}
                </div>
              </div>
            );
          })}
          {isLoading &&
            (messages.length === 0 ||
              messages[messages.length - 1].type === "human") && (
              <div className="flex items-start gap-3 mt-3">
                {" "}
                {/* AI message row structure */}
                <div className="relative group max-w-[85%] md:max-w-[80%] rounded-xl p-3 shadow-sm break-words bg-neutral-800 text-neutral-100 rounded-bl-none w-full min-h-[56px]">
                  {liveActivityEvents.length > 0 ? (
                    <div className="text-xs">
                      <ActivityTimeline
                        processedEvents={liveActivityEvents}
                        isLoading={true}
                      />
                    </div>
                  ) : (
                    <div className="flex items-center justify-start h-full">
                      <Loader2 className="h-5 w-5 animate-spin text-neutral-400 mr-2" />
                      <span>Processing...</span>
                    </div>
                  )}
                </div>
              </div>
            )}
        </div>
      </ScrollArea>
      {screenshot && (
        <div className="flex flex-col items-center mt-4 p-4 bg-neutral-900 border-t border-neutral-700">
          <div className="flex justify-between items-center w-full max-w-2xl mb-2">
            <h3 className="text-sm font-medium text-neutral-300">Source Content</h3>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => setScreenshot(null)}
              className="text-neutral-400 hover:text-neutral-200"
            >
              Close
            </Button>
          </div>
          <img 
            src={`data:image/png;base64,${screenshot}`} 
            alt="Source Screenshot" 
            className="max-w-full max-h-96 border border-neutral-600 rounded-lg shadow-lg"
          />
        </div>
      )}
      <InputForm
        onSubmit={onSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={messages.length > 0}
      />
    </div>
  );
}
