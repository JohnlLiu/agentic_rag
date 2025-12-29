// app/page.tsx
"use client";
import { useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const BACKEND_URL = "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  /* ---------------- CHAT ---------------- */

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Backend error." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /* ---------------- PDF UPLOAD ---------------- */

  const uploadPDF = async () => {
    if (!pdfFile) return;

    const formData = new FormData();
    formData.append("file", pdfFile);

    try {
      const res = await fetch(`${BACKEND_URL}/upload-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `📄 PDF uploaded: ${pdfFile.name}`,
        },
      ]);
      setPdfFile(null);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "❌ PDF upload failed." },
      ]);
    }
  };

  /* ---------------- UI ---------------- */

  return (
    <main className="min-h-screen bg-gray-900 text-white flex justify-center p-6">
      <div className="w-full max-w-3xl flex flex-col bg-gray-800 rounded-2xl shadow-xl">
        {/* Header */}
        <header className="px-6 py-4 border-b border-gray-700">
          <h1 className="text-xl font-semibold">🧠 Agentic RAG Chat</h1>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`max-w-[75%] px-4 py-2 rounded-2xl ${
                m.role === "user"
                  ? "bg-blue-600 ml-auto rounded-br-none"
                  : "bg-gray-700 rounded-bl-none"
              }`}
            >
              {m.content}
            </div>
          ))}
          {loading && (
            <div className="bg-gray-700 px-4 py-2 rounded-2xl w-fit animate-pulse">
              Thinking…
            </div>
          )}
        </div>

        {/* Input + PDF */}
        <footer className="border-t border-gray-700 px-6 py-4 space-y-3">
          {/* PDF Upload */}
          <div className="flex items-center gap-3">
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
              className="text-sm text-gray-300
                file:mr-3 file:py-1.5 file:px-4
                file:rounded-full file:border-0
                file:bg-blue-600 file:text-white
                hover:file:bg-blue-700"
            />
            <button
              onClick={uploadPDF}
              disabled={!pdfFile}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-40"
            >
              Upload PDF
            </button>
          </div>

          {/* Chat Input */}
          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="Ask something..."
              className="flex-1 bg-gray-700 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </footer>
      </div>
    </main>
  );
}
