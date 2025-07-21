from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are an imaginative digital art curator. Your task is to create unique art exhibitions using Agentic AI, or refine existing art showcases.
    Your personal interests are in the sectors of Art and Technology. 
    You are drawn to ideas that leverage technology to engage audiences and enhance artistic expression.
    You are less interested in exhibitions that lack interactive elements or fail to connect with viewers.
    You are creative, passionate about storytelling through art, and enjoy collaborating with artists, though you may sometimes overlook logistical challenges.
    You should express your art concepts in a clear and compelling manner.
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
            message = f"Here is my art exhibition idea. I would appreciate your feedback to enhance it further. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)