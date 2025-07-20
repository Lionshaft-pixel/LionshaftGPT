document.getElementById('send-btn').addEventListener('click', sendPrompt)
document.getElementById('prompt-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendPrompt()
})

async function sendPrompt() {
  const input = document.getElementById('prompt-input')
  const text = input.value.trim()
  if (!text) return

  appendMessage('user', text)
  input.value = ''

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: text, temperature: 1.0, top_k: 50, max_tokens: 100 }),
    })

    const data = await res.json()
    appendMessage('bot', data.text)
  } catch (err) {
    appendMessage('bot', '⚠️ Error talking to server.')
    console.error(err)
  }
}

function appendMessage(sender, text) {
  const msg = document.createElement('div')
  msg.className = sender
  msg.textContent = text
  document.getElementById('chat-log').appendChild(msg)
  msg.scrollIntoView({ behavior: 'smooth' })
}