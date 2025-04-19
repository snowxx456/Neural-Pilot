// pages/api/search-datasets.js

export default async function handler(req, res) {

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { query } = req.body;

  try {
    const pythonResponse = await fetch(
      `${process.env.NEXT_PUBLIC_SERVER_URL}/api/search/`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      }
    );

    if (!pythonResponse.ok) {
      throw new Error("Backend service error");
    }

    const data = await pythonResponse.json();
    res.status(200).json(data);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Failed to fetch datasets" });
  }
}
