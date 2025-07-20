import React, { useRef, useEffect, useState } from "react"
import { MessageBubble } from "./MessageBubble"
import { Input } from "./ui/input"
import { Button } from "./ui/button"

interface ChatMessage {
  user: string
  bot: string
}

export function ChatWindow() {
  const [prompt, setPrompt] = useState("")
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<ChatMessage[]>([])
  const containerRef = useRef<HTMLDivElement>(null)

  const sendPrompt = async () => {
    const trimmed = prompt.trim()
    if (!trimmed) return

    setPrompt("")
    setLoading(true)
    setHistory((prev: ChatMessage[]) => [...prev, { user: trimmed, bot: "" }])

    // Fake response
    setTimeout(() => {
      setHistory((prev: ChatMessage[]) =>
        prev.map((msg, i) =>
          i === prev.length - 1 ? { ...msg, bot: `Echo: ${trimmed}` } : msg
        )
      )
      setLoading(false)
    }, 1000)
  }

  useEffect(() => {
    containerRef.current?.scrollTo({ top: containerRef.current.scrollHeight, behavior: "smooth" })
  }, [history])

  return (
    <div className="flex flex-col h-screen p-4 bg-gradient-to-r from-gray-900 to-black text-white">
      <div ref={containerRef} className="flex-1 overflow-auto space-y-2 pb-4 pr-2">
        {history.map((msg, i) => (
          <React.Fragment key={i}>
            <MessageBubble text={msg.user} isUser />
            <MessageBubble text={msg.bot} />
          </React.Fragment>
        ))}
        {loading && <MessageBubble text="Typing..." />}
      </div>

      <div className="flex items-center space-x-2">
        <Input
          disabled={loading}
          placeholder="Type your message..."
          value={prompt}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPrompt(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === "Enter" && sendPrompt()}
        />
        <Button onClick={sendPrompt} disabled={loading || !prompt.trim()}>
          {loading ? "Sending..." : "Send"}
        </Button>
      </div>
    </div>
  )
}