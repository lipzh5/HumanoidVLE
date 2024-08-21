# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import asyncio
from subrouter import SubRouter


async def run_sub_router():
    sub_router = SubRouter()
    loop = asyncio.get_event_loop()
    print(f'task loop is running!!!')
    task1 = loop.create_task(sub_router.sub_vcap_data())
    task2 = loop.create_task(sub_router.route_vle_task())
    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    asyncio.run(run_sub_router())
   