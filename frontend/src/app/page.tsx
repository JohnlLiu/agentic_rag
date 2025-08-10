// app/page.tsx
"use client";
import { useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState("");
  const [pdfFile, setPdfFile] = useState<File | null>(null);

  const sendMessage = async () => {
    if (!input.trim() && !pdfFile) return;

    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");

    // Example backend call — replace with your API endpoint
    const formData = new FormData();
    formData.append("message", input);
    if (pdfFile) formData.append("pdf", pdfFile);

    const res = await fetch("/api/chat", { method: "POST", body: formData });
    const data = await res.json();

    setMessages([...newMessages, { role: "assistant", content: data.reply }]);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center py-10">
      <h1 className="text-3xl font-bold mb-6">💬 Chat with AI</h1>

      <div className="w-full max-w-2xl bg-gray-800 rounded-lg shadow-lg p-4 flex flex-col space-y-4">
        {/* Chat messages */}
        <div className="flex-1 overflow-y-auto max-h-[500px] space-y-3">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`p-3 rounded-lg max-w-[80%] ${
                m.role === "user" ? "bg-blue-600 ml-auto" : "bg-gray-700"
              }`}
            >
              {m.content}
            </div>
          ))}
        </div>

        {/* PDF upload */}
        <div className="flex items-center gap-2">
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
            className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4
                       file:rounded-full file:border-0 file:text-sm file:font-semibold
                       file:bg-blue-500 file:text-white hover:file:bg-blue-600"
          />
          {pdfFile && <p className="text-xs text-gray-400">{pdfFile.name}</p>}
        </div>

        {/* Input + Send */}
        <div className="flex items-center gap-2">
          <input
            className="flex-1 p-3 rounded-lg bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            onClick={sendMessage}
            className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
