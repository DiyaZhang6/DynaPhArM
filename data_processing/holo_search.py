import pandas as pd
import requests
import time
import json
import os

# --- 用户配置区域 ---
INPUT_CSV_FILE = '/home/zdy/Project2/data/file.csv'  # 您的输入CSV文件路径
DRUG_COLUMN_IDENTIFIER = 'name'  # 包含药物名称的列名
# 新的 holo_id 将直接写入此文件

# --- PDB API 配置 ---
PDB_SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
REQUEST_TIMEOUT = 30  # 秒
DELAY_BETWEEN_REQUESTS = 0.75  # 秒 - API调用之间的延迟，非常重要！


# --- API请求辅助函数 (已修正和优化) ---
def make_pdb_api_request(query_json):
    """
    向PDB Search API发送POST请求并处理响应。
    返回解析后的JSON数据或None（如果出错）。
    """
    response_obj = None  # 初始化response_obj，确保其在后续的except块中可用
    try:
        response_obj = requests.post(PDB_SEARCH_URL, json=query_json, timeout=REQUEST_TIMEOUT)
        time.sleep(DELAY_BETWEEN_REQUESTS)
        response_obj.raise_for_status()  # 对4xx或5xx错误抛出HTTPError

        if response_obj.status_code == 204:  # 无内容
            return None
        return response_obj.json()
    except requests.exceptions.HTTPError as http_err:  # http_err仅在此块中定义
        status_code = "N/A"
        response_text = "无响应体"
        if http_err.response is not None:
            status_code = http_err.response.status_code
            response_text = http_err.response.text[:200]

        if status_code == 404:
            # 对于404 (未找到)，可以不打印详细错误，因为它可能在预期之内
            pass
        else:
            print(f"  错误：HTTPError (状态码 {status_code}): {http_err} - 响应: {response_text}")
        return None
    except requests.exceptions.JSONDecodeError as json_decode_err:  # json_decode_err仅在此块中定义
        status_code_info = "N/A"
        content_type_info = "N/A"
        response_text_info = "N/A (response_obj不可用或之前已出错)"

        if response_obj is not None:  # 使用在try块中定义的response_obj
            status_code_info = response_obj.status_code
            content_type_info = response_obj.headers.get('Content-Type', 'N/A')
            response_text_info = response_obj.text[:500]

        print(f"  错误：JSONDecodeError (状态码 {status_code_info}, 内容类型 {content_type_info}): {json_decode_err}")
        print(f"    未能解析的响应文本 (前500字符):\n'''\n{response_text_info}\n'''")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"  错误：请求超时: {timeout_err}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"  错误：连接错误: {conn_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"  错误：RequestException (通用网络请求): {req_err}")
        return None
    except Exception as e:
        print(f"  API请求中发生未知错误: {e}")
        return None


# 请只替换您脚本中的 find_comp_ids_for_drug_name 函数
# 其他函数 (make_pdb_api_request, find_pdb_entries_with_comp_id, search_pdb_for_drug, main)
# 以及顶部的常量定义保持不变。

# 请只替换您脚本中的 find_comp_ids_for_drug_name 函数
# 其他函数 (make_pdb_api_request, find_pdb_entries_with_comp_id, search_pdb_for_drug, main)
# 以及顶部的常量定义保持不变。

def find_comp_ids_for_drug_name(drug_name_upper):
    """
    根据药物名称搜索PDB化学组分词典，返回匹配的comp_id列表。
    (已更新 request_options 中的分页参数并移除排序)
    """
    print(f"    正在为药物名称 '{drug_name_upper}' 搜索化学组分ID(comp_id) (使用 chemical service)...")
    comp_ids_found = set()
    query_json = {
        "query": {
            "type": "terminal",
            "service": "chemical",
            "parameters": {
                "value": drug_name_upper,
                "type": "name",  # 指明value的类型是名称
                "match_type_name": "exact_match"  # 指定名称的匹配方式为精确匹配
                # 如果效果不佳，后续可以尝试 "contains_phrase"
            }
        },
        "return_type": "chem_comp",  # 我们希望返回化学组分ID
        "request_options": {
            "paginate": {"start": 0, "rows": 10}  # <<< --- 【重要修正】使用 "paginate" 并暂时移除 sort
        }
    }

    results = make_pdb_api_request(query_json)  # 调用您脚本中已有的make_pdb_api_request

    if results and "result_set" in results and results["result_set"]:
        for item in results["result_set"]:
            comp_id_val = None
            if isinstance(item, str):
                comp_id_val = item
            elif isinstance(item, dict) and 'identifier' in item:
                comp_id_val = str(item['identifier'])

            if comp_id_val:
                comp_ids_found.add(comp_id_val.upper())
        print(
            f"      找到与 '{drug_name_upper}' (使用chemical service) 相关的 comp_id(s): {comp_ids_found if comp_ids_found else '无'}")
    else:
        print(f"      未能通过chemical service为 '{drug_name_upper}' 找到任何化学组分ID (API调用可能失败或无结果)。")

    return list(comp_ids_found)


def find_pdb_entries_with_comp_id(comp_id):
    """
    根据给定的comp_id，搜索包含该配体的PDB条目ID。
    """
    print(f"      正在为化学组分ID '{comp_id}' 搜索PDB条目...")
    pdb_entries_found = set()
    query_json = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity_instance_container_identifiers.comp_id",
                "operator": "exact_match",
                "value": comp_id.upper()
            }
        },
        "return_type": "entry",
        "request_options": {
            "pager": {"start": 0, "rows": 100},  # 一个配体可能存在于多个PDB中
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "results_content_type": ["experimental"]
        }
    }

    results = make_pdb_api_request(query_json)
    if results and "result_set" in results and results["result_set"]:
        for item in results["result_set"]:
            pdb_id_val = None
            if isinstance(item, str):
                pdb_id_val = item
            elif isinstance(item, dict) and 'identifier' in item:
                pdb_id_val = str(item['identifier'])

            if pdb_id_val:
                pdb_entries_found.add(pdb_id_val.upper())
        print(f"        为 comp_id '{comp_id}' 找到 {len(pdb_entries_found)} 个PDB条目。")
    else:
        print(f"        未能为 comp_id '{comp_id}' 找到任何PDB条目。")

    return list(pdb_entries_found)


def search_pdb_for_drug(drug_name):  # 重写此函数以实现新的搜索逻辑
    """
    为给定的药物名称查找真实结合的holo PDB ID。
    """
    if not drug_name or pd.isna(drug_name):
        return ""
    drug_name_upper = str(drug_name).strip().upper()
    if not drug_name_upper:
        return ""

    # 步骤1: 根据药物名称查找化学组分ID (comp_id)
    comp_ids = find_comp_ids_for_drug_name(drug_name_upper)

    if not comp_ids:
        print(f"    由于未能找到 '{drug_name}' 的comp_id，无法进行精确的holo结构搜索。")
        return ""

    all_holo_pdb_ids_for_drug = set()
    # 步骤2: 为每个找到的comp_id查找包含它的PDB条目
    for comp_id in comp_ids:
        # 过滤掉一些非常常见的、不太可能是目标药物的comp_id（可根据需要调整）
        # 例如，水(HOH)，硫酸根(SO4)，磷酸根(PO4)，甘油(GOL)，乙二醇(EDO)等。
        # 但如果药物名称本身就是这些，则不应过滤。
        common_filter = {"HOH", "SO4", "PO4", "GOL", "EDO", "NA", "K", "CL", "MG", "CA", "ZN"}
        if comp_id in common_filter and drug_name_upper != comp_id:  # 如果药物名本身不是这个通用comp_id
            print(f"      跳过常见的化学组分ID: {comp_id} (除非它与药物名完全匹配)")
            continue

        pdb_entries = find_pdb_entries_with_comp_id(comp_id)
        if pdb_entries:
            all_holo_pdb_ids_for_drug.update(pdb_entries)

    if all_holo_pdb_ids_for_drug:
        return ",".join(sorted(list(all_holo_pdb_ids_for_drug)))
    else:
        # print(f"    药物 '{drug_name}' (comp_ids: {comp_ids}) 未能找到任何包含这些特定配体的PDB条目。")
        return ""


def main():
    base_dir = os.getcwd()
    input_csv_abs_path = os.path.join(base_dir, INPUT_CSV_FILE)  # 使用INPUT_CSV_FILE
    try:
        input_csv_abs_path = os.path.abspath(INPUT_CSV_FILE)  # 使用INPUT_CSV_FILE
        base_dir = os.path.dirname(input_csv_abs_path)
    except Exception:
        print(f"警告：无法确定 '{INPUT_CSV_FILE}' 的绝对路径，输出文件将保存在当前工作目录 '{base_dir}'。")

    print(f"开始处理文件: {input_csv_abs_path}")
    print(f"药物名称将从列标识符 '{DRUG_COLUMN_IDENTIFIER}' 读取。")
    print(f"将在原文件 '{input_csv_abs_path}' 中直接添加/更新名为 'holo_id' 的列。")
    print(f"将使用两步搜索策略：药物名->comp_id->包含该comp_id的PDB条目。")
    print("-" * 30)

    try:
        df = pd.read_csv(input_csv_abs_path, skipinitialspace=True)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_csv_abs_path}' 未找到。")
        return
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{input_csv_abs_path}' 为空。")
        return
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
        return

    if DRUG_COLUMN_IDENTIFIER not in df.columns:
        print(f"错误: 在CSV文件的表头中未找到药物名称列 '{DRUG_COLUMN_IDENTIFIER}'。")
        print(f"可用的列名有: {df.columns.tolist()}")
        return

    # 创建一个空列表来存储新生成的holo_id数据
    new_holo_id_column_data = []
    total_rows = len(df)
    print(f"共找到 {total_rows} 行药物条目需要处理。\n")

    for index, row in df.iterrows():
        drug_name_from_row = row[DRUG_COLUMN_IDENTIFIER]
        actual_drug_name_to_search = ""
        if pd.notna(drug_name_from_row) and str(drug_name_from_row).strip():
            actual_drug_name_to_search = str(drug_name_from_row).strip()

        print(f"正在处理CSV行 {index + 1}/{total_rows}: 药物='{actual_drug_name_to_search}'...")

        if not actual_drug_name_to_search:
            print("  药物名称为空或无效，此行的holo_id将设置为空。")
            new_holo_id_column_data.append("")
            continue

        pdb_ids_str = search_pdb_for_drug(actual_drug_name_to_search)
        new_holo_id_column_data.append(pdb_ids_str)

        if pdb_ids_str:
            print(f"  为药物 '{actual_drug_name_to_search}' 找到的结合PDB ID(s): {pdb_ids_str}")
        else:
            print(f"  未能为药物 '{actual_drug_name_to_search}' 找到任何精确结合的PDB ID。")

    df['holo_id'] = new_holo_id_column_data

    try:
        df.to_csv(input_csv_abs_path, index=False, encoding='utf-8-sig')
        print("-" * 30)
        print(f"\n处理完成! ✨")
        print(f"已在原文件 '{input_csv_abs_path}' 中成功更新/添加了 'holo_id' 列。")
    except Exception as e:
        print(f"写入更新后的CSV文件 '{input_csv_abs_path}' 时发生错误: {e}")


if __name__ == '__main__':
    # 将 main() 函数重命名为 process_file() 以匹配之前脚本的结构，或者直接调用 main()
    # 为了保持与您错误信息中提到的 process_file 的一致性，如果您之前的脚本是这样调用的
    # 此处我们直接定义并调用 main()
    main()