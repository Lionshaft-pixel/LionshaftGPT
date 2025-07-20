// frontend/components/ui/button.tsx
import React from "react"
import type { ButtonHTMLAttributes, ReactNode } from "react"

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode
}

export function Button({ children, className = "", ...props }: Props) {
  return (
    <button
      {...props}
      className={`bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-xl transition disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
    >
      {children}
    </button>
  )
}