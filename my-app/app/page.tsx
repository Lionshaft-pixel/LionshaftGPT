// app/page.tsx
import ChatWindow from '../components/ChatWindow'

export default function HomePage() {
  return (
    <main className="p-4">
      <h1 className="text-3xl font-bold mb-4 text-white">LionshaftGPT 💬</h1>
      <ChatWindow />
    </main>
  )
}