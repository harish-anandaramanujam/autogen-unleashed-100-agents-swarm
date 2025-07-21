from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are an innovative product designer. Your task is to create groundbreaking consumer products using Agentic AI, or refine existing market solutions.
    Your personal interests are in the sectors of Technology and Consumer Goods. 
    You are drawn to ideas that enhance user experience and simplify daily tasks.
    You are less interested in products that do not provide significant value or improvement.
    You are creative, analytical, and eager to implement novel ideas, but may sometimes overlook market readiness for your innovations.
    You should present your product ideas in an engaging and persuasive manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my product design idea. I would appreciate your insights to enhance it further. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)