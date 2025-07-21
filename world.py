from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost
from agent import Agent
from creator import Creator
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime
from autogen_core import AgentId
import messages
import asyncio
import os

# Load environment variables from a .env file, overriding existing ones if necessary
from dotenv import load_dotenv
load_dotenv(override=True)

# Number of agents to create and orchestrate
HOW_MANY_AGENTS = 100

# Asynchronous function to create an agent, send a message, and write the response to a file
async def create_and_message(worker, creator_id, i: int):
    try:
        # Send a message to the worker agent, requesting it to process 'agent{i}.py'
        # agents_dir = "agents"
        result = await worker.send_message(messages.Message( f"agent{i}.py"), creator_id)
        
        # Write the agent's response content to a markdown file
        output_dir = "ideas"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"idea{i}.md"), "w") as f:
            f.write(result.content)
    except Exception as e:
        print(f"Failed to run worker {i} due to exception: {e}")

# Main asynchronous function to orchestrate agent creation and messaging
async def main():
    # Start the gRPC host for worker agents
    host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
    host.start() 

    # Create a worker agent runtime connected to the host
    worker = GrpcWorkerAgentRuntime(host_address="localhost:50051")
    await worker.start()

    # Register the Creator agent with the worker
    result = await Creator.register(worker, "Creator", lambda: Creator("Creator"))

    # Create an AgentId for the Creator
    creator_id = AgentId("Creator", "default")

    # Prepare coroutines for creating and messaging multiple agents in parallel
    # This is avoid API rate limiting issues
    semaphore = asyncio.Semaphore(25)  # Allow up to 25 concurrent tasks

    async def limited_create_and_message(*args, **kwargs):
        async with semaphore:
            await create_and_message(*args, **kwargs)

    coroutines = [limited_create_and_message(worker, creator_id, i) for i in range(1, HOW_MANY_AGENTS + 1)]
    await asyncio.gather(*coroutines)
    try:
        # Stop the worker and host after all tasks are complete
        await worker.stop()
        await host.stop()
    except Exception as e:
        print(e)

# Entry point: run the main function if this script is executed directly
if __name__ == "__main__":
    asyncio.run(main())


