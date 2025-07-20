import React from "react"

export function MessageBubble({ text, isUser }: { text: string; isUser?: boolean }) {
  return (
    <div className={`p-4 max-w-md my-2 rounded-2xl shadow ${isUser ? "self-end bg-blue-600 text-white" : "self-start bg-gray-700 text-gray-100"}`}>
      {text}
    </div>
  )
}