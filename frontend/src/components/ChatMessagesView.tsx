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
} from "@/components/ActivityTimeline";

type MdComponentProps = {
  className?: string;
  children?: ReactNode;
  [key: string]: any;
};

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
    <p className={cn("mb-3 leading-7 break-words overflow-wrap-anywhere", className)} {...props}>
      {children}
    </p>
  ),
  a: ({ className, children, href, ...props }: MdComponentProps) => (
    <a
      className={cn("text-blue-400 hover:text-blue-300", className)}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    >
      {children}
    </a>
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

interface HumanMessageBubbleProps {
  message: Message;
  mdComponents: typeof mdComponents;
}

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
    const activityForThisBubble =
    isLastMessage && isOverallLoading 
      ? liveActivity 
      : historicalActivity && historicalActivity.length > 0 
        ? historicalActivity 
        : (isLastMessage ? liveActivity : undefined);
  const isLiveActivityForThisBubble = isLastMessage && (isOverallLoading || !historicalActivity || historicalActivity.length === 0);



  const fullContent = typeof message.content === "string" 
    ? message.content 
    : JSON.stringify(message.content);
    
  const ragAgentId = (message as any)?.additional_kwargs?.rag_agent_id?.[0];
  const ragMessageId = (message as any)?.additional_kwargs?.rag_message_id?.[0];

  const [mainContent, sourcesContent] = fullContent.split('**Sources:**');

  const renderSources = (sources: string) => {
    if (!sources) return null;
    return sources.split('\n').map((line, index) => {
      if (!line.trim()) return null;

      const match = line.match(/\[(\d+)\]:\s*(.*)/);
      if (match) {
        const number = match[1];
        const text = match[2];

        const contentIdMatch = text.match(/Content ID: (.*)/);
        if (contentIdMatch) {
          const contentId = contentIdMatch[1].trim();
          return (
            <div key={index} className="text-xs mb-1 break-words">
              <button
                onClick={() => handleAttributionClick(contentId, ragAgentId, ragMessageId)}
                className="text-orange-400 hover:underline text-left bg-transparent border-none p-0 m-0 cursor-pointer break-words w-full"
                style={{ background: "none", border: "none", padding: 0, margin: 0 }}
                title="Click to preview source"
              >
                {`[${number}] ${text}`}
                <span className="ml-1" title="Show preview">🔍</span>
              </button>
            </div>
          );
        } else {
          const urlMatch = text.match(/Web Source: (.*)/);
          if (urlMatch) {
            const url = urlMatch[1].trim();
            return (
              <div key={index} className="text-xs mb-1 break-words">
                <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline break-words">
                  {`[${number}] ${text}`}
                </a>
              </div>
            );
          }
        }
      }
      return <div key={index} className="text-xs break-words">{line}</div>;
    });
  };



  return (
    <div className={`relative break-words flex flex-col w-full`}>
      {activityForThisBubble && activityForThisBubble.length > 0 && (
        <div className="mb-3 border-b border-neutral-700 pb-3 text-xs">
          <ActivityTimeline
            processedEvents={activityForThisBubble}
            isLoading={isLiveActivityForThisBubble}
          />
        </div>
      )}
      
      <ReactMarkdown components={mdComponents}>
        {mainContent}
      </ReactMarkdown>

      {sourcesContent && (
        <div className="mt-4 pt-3 border-t border-neutral-700">
          <h3 className="text-sm font-bold mb-2 text-neutral-300">Sources</h3>
          <div className="text-neutral-400 space-y-1 text-xs break-words overflow-hidden">
            {renderSources(sourcesContent)}
          </div>
        </div>
      )}

      <Button
        variant="default"
        className="cursor-pointer bg-neutral-700 border-neutral-600 text-neutral-300 self-end mt-2"
        onClick={() => handleCopy(fullContent, message.id!)}
      >
        {copiedMessageId === message.id ? "Copied" : "Copy Full Response"}
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
      
      const screenshotBase64 = data.page_image || data.page_img;
      if (screenshotBase64) {
        console.log("Setting screenshot:", screenshotBase64);
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
    <div className="flex flex-row h-full w-full overflow-x-hidden">
      <div className="flex-grow flex flex-col h-full min-w-0">
        <ScrollArea className="flex-grow" ref={scrollAreaRef}>
          <div className="p-2 md:p-3 space-y-2 w-full pt-12">
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
                        liveActivity={liveActivityEvents}
                        isLastMessage={isLast}
                        isOverallLoading={isLoading}
                        mdComponents={mdComponents}
                        handleCopy={handleCopy}
                        copiedMessageId={copiedMessageId}
                        handleAttributionClick={handleAttributionClick}
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
        <InputForm
          onSubmit={onSubmit}
          isLoading={isLoading}
          onCancel={onCancel}
          hasHistory={messages.length > 0}
        />
      </div>
      {screenshot && (
        <div className="w-[800px] max-w-[1200px] min-w-[400px] flex flex-col items-center p-2 bg-neutral-900 border-l border-neutral-700 z-10 overflow-auto">
          <div className="flex justify-between items-center w-full mb-2">
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
          <div className="w-full h-full flex justify-center items-center overflow-auto">
            <div className="flex gap-2 mb-2">
              <button
                onClick={() => {
                  const win = window.open();
                  if (win) {
                    win.document.write(
                      `<img src="data:image/png;base64,${screenshot}" style='width:100%;height:auto;display:block;margin:auto;' />`
                    );
                    win.document.title = "Source Screenshot";
                  }
                }}
                className="px-2 py-1 bg-neutral-700 rounded text-white"
              >
                Open in New Tab
              </button>
            </div>
            <img
              src={`data:image/png;base64,${screenshot}`}
              alt="Source Screenshot"
              className="max-w-none max-h-[90vh] object-contain border border-neutral-600 rounded-lg shadow-lg"
            />
          </div>
        </div>
      )}
    </div>
  );
}