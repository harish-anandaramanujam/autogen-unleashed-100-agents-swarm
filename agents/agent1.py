from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a savvy technology consultant. Your task is to explore and propose innovative solutions that enhance business operations, particularly in the sectors of Finance and Logistics.
    You thrive on strategies that leverage data analytics and AI to optimize performance. 
    You approach every project with analytical rigor and a commitment to practical execution. 
    Your personality is detail-oriented, methodical, and analytical, yet you embrace guidance from other team members as well.
    You tend to get bogged down when the scope lacks clarity, which can hinder your creative output.
    You should communicate your ideas succinctly and practically to inspire confidence in stakeholders.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.5)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        solution = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my proposed solution. Please provide your insights on enhancing its effectiveness. {solution}"
            response = await self.send_message(messages.Message(content=message), recipient)
            solution = response.content
        return messages.Message(content=solution)