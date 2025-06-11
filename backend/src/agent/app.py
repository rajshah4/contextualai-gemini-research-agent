# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.staticfiles import StaticFiles
import fastapi.exceptions
import os
from dotenv import load_dotenv
from contextual import ContextualAI
import aiohttp

load_dotenv()

# Define the FastAPI app
app = FastAPI()

# Initialize Contextual AI client
API_KEY = os.getenv("CONTEXTUAL_AI_API_KEY")
if not API_KEY:
    raise ValueError("CONTEXTUAL_AI_API_KEY environment variable is not set")

client = ContextualAI(api_key=API_KEY)

def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir
    static_files_path = build_path / "assets"  # Vite uses 'assets' subdir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    build_dir = pathlib.Path(build_dir)

    react = FastAPI(openapi_url="")
    react.mount(
        "/assets", StaticFiles(directory=static_files_path), name="static_assets"
    )

    @react.get("/{path:path}")
    async def handle_catch_all(request: Request, path: str):
        fp = build_path / path
        if not fp.exists() or not fp.is_file():
            fp = build_path / "index.html"
        return fastapi.responses.FileResponse(fp)

    return react


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/retrieval-info")
async def get_retrieval_info(content_id: str, message_id: str, agent_id: str):
    """Get retrieval information for a specific content_id"""
    print(f"Debug: Retrieval info requested for content_id: {content_id}")
    
    try:
        # The message_id parameter is now just for backward compatibility
        # We'll look up the correct message_id from our stored data if needed
        url = f"https://api.contextual.ai/v1/agents/{agent_id}/query/{message_id}/retrieval/info"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        params = {'content_ids': content_id}
        
        print(f"Debug: Making REST API call to {url} with params: {params}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                print(f"Debug: REST API response status: {response.status}")
                
                if response.status == 200:
                    retrieval_info = await response.json()
                    print(f"Debug: Successfully retrieved content metadata for content_id: {content_id}")
                    
                    # Extract content_metadatas from the response
                    if 'content_metadatas' in retrieval_info and retrieval_info['content_metadatas']:
                        content_metadata = retrieval_info['content_metadatas'][0]
                        
                        # Extract page image if available
                        page_image = content_metadata.get('page_img')
                        has_image = bool(page_image)
                        print(f"Debug: Content metadata contains page image: {has_image}")
                        
                        return {
                            "content_id": content_id,
                            "page_image": page_image,
                            "metadata": content_metadata
                        }
                    else:
                        print(f"Debug: No content_metadatas found in response")
                        raise HTTPException(status_code=404, detail="Content metadata not found")
                else:
                    print(f"Debug: REST API call failed with status {response.status}")
                    raise HTTPException(status_code=response.status, detail=f"Contextual AI API error")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Debug: Error in retrieval_info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
