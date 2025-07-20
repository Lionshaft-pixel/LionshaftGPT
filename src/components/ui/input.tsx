// frontend/components/ui/input.tsx
import React from "react"
import type { InputHTMLAttributes } from "react"

type Props = InputHTMLAttributes<HTMLInputElement>

export function Input({ className = "", ...props }: Props) {
  return (
    <input
      {...props}
      className={`w-full p-2 rounded-xl bg-gray-800 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 ${className}`}
    />
  )
}